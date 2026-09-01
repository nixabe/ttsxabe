//! Just enough of the pickle virtual machine to rebuild a state dict.
//!
//! # Why this is not "running pickle"
//!
//! Unpickling in general executes whatever the stream names: it imports
//! modules, calls constructors and rebuilds arbitrary object graphs. That is
//! what makes a pre-1.6 torch file unreadable without the model's own class
//! definitions, and it is the line this crate does not cross.
//!
//! A **state dict** needs none of it. It is a mapping of strings to tensors,
//! and the only three things its stream ever names are `collections.OrderedDict`,
//! `torch._utils._rebuild_tensor_v2` and a storage class. This machine
//! implements those three and refuses every other [`Opcode::Global`] by name,
//! so a checkpoint that would need real code execution fails with a sentence
//! instead of being half-read.
//!
//! # What is (and isn't) here
//!
//! The opcodes torch's writer emits at protocol 2, plus the handful protocols 3
//! and 4 add, so that a checkpoint saved by a newer torch still opens. Anything
//! else is an error naming the byte and the mnemonic, because the fix is always
//! to implement it rather than to guess what it would have pushed.
//!
//! Tensor *data* is never touched here. A [`Value::Tensor`] records which
//! storage it borrows from and where; resolving that to bytes is
//! [`crate::file`]'s job.
//!
//! Start at [`load`].

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::PtError;

/// Element type of a storage, named by the class the pickle imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `torch.FloatStorage`.
    F32,
    /// `torch.HalfStorage`.
    F16,
    /// `torch.BFloat16Storage`. Turing has no bf16 arithmetic, so it is widened
    /// on read - which is exact, since bf16 is the top half of an f32.
    Bf16,
}

impl Dtype {
    /// Maps a storage class name onto an element type, or `None` if this reader
    /// does not decode it.
    fn parse(class: &str) -> Option<Self> {
        match class {
            "torch.FloatStorage" => Some(Self::F32),
            "torch.HalfStorage" => Some(Self::F16),
            "torch.BFloat16Storage" => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Width of one element in bytes.
    pub const fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// The name for error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// A storage: one flat run of elements, stored as its own archive entry.
///
/// Several tensors may name the same storage - that is how torch represents
/// shared parameters - so this is cheap to clone and carries no data.
#[derive(Debug, Clone)]
pub struct Storage {
    /// The storage's key, which is also its file name under `data/`.
    pub key: String,
    /// Element type.
    pub dtype: Dtype,
    /// Elements the storage holds.
    pub numel: usize,
}

/// A tensor: a window onto a storage, with a shape and a stride.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Where its elements come from.
    pub storage: Storage,
    /// First element of its window, in elements.
    pub offset: usize,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Stride per dimension, in elements. Checked against `shape` by the
    /// caller: a non-contiguous view cannot be borrowed.
    pub stride: Vec<usize>,
}

/// A mapping's entries, shared because pickle mutates one after memoising it.
type Entries = Rc<RefCell<Vec<(Value, Value)>>>;

/// Anything the machine can put on its stack.
///
/// Dictionaries and lists are shared handles because pickle mutates them after
/// they are memoised: `EMPTY_DICT`, `BINPUT`, then `SETITEMS` fills in the same
/// object a later `BINGET` may hand back. Copying by value here would return
/// the empty one.
#[derive(Debug, Clone)]
pub enum Value {
    /// `None`.
    None,
    /// A boolean.
    ///
    /// Carried rather than read: nothing in a state dict is a bool, but the
    /// machine still has to push what the opcode says.
    #[allow(dead_code, reason = "pushed by the stream, never inspected")]
    Bool(bool),
    /// An integer, of any pickled width.
    Int(i64),
    /// A float. Carried rather than read, as [`Value::Bool`] is.
    #[allow(dead_code, reason = "pushed by the stream, never inspected")]
    Float(f64),
    /// A string.
    Str(Rc<str>),
    /// A byte string. Carried rather than read, as [`Value::Bool`] is.
    #[allow(dead_code, reason = "pushed by the stream, never inspected")]
    Bytes(Rc<[u8]>),
    /// A tuple.
    Tuple(Rc<[Value]>),
    /// A list.
    List(Rc<RefCell<Vec<Value>>>),
    /// A mapping, in insertion order. Linear rather than hashed: the mappings
    /// in a checkpoint are either tiny or read exactly once.
    Dict(Entries),
    /// An imported name that has not been called.
    Global {
        /// Module it was imported from.
        module: Rc<str>,
        /// Attribute name.
        name: Rc<str>,
    },
    /// A rebuilt tensor.
    Tensor(Rc<Tensor>),
    /// A storage, resolved from a persistent id.
    Storage(Storage),
}

impl Value {
    /// A one-word description, for errors that report what was found.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bool(_) => "a bool",
            Self::Int(_) => "an int",
            Self::Float(_) => "a float",
            Self::Str(_) => "a string",
            Self::Bytes(_) => "bytes",
            Self::Tuple(_) => "a tuple",
            Self::List(_) => "a list",
            Self::Dict(_) => "a dict",
            Self::Global { .. } => "an imported name",
            Self::Tensor(_) => "a tensor",
            Self::Storage(_) => "a storage",
        }
    }

    /// The entries of a mapping, or `None` if this is not one.
    pub fn as_dict(&self) -> Option<std::cell::Ref<'_, Vec<(Value, Value)>>> {
        match self {
            Self::Dict(d) => Some(d.borrow()),
            _ => None,
        }
    }

    /// Looks a string key up in a mapping.
    pub fn get(&self, key: &str) -> Option<Value> {
        let entries = self.as_dict()?;
        entries
            .iter()
            .find(|(k, _)| matches!(k, Value::Str(s) if &**s == key))
            .map(|(_, v)| v.clone())
    }

    /// The integer this holds, if it is one.
    fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Int(v) if *v >= 0 => Some(*v as usize),
            _ => None,
        }
    }

    /// The elements of a tuple or list, if this is one.
    fn as_seq(&self) -> Option<Vec<Value>> {
        match self {
            Self::Tuple(t) => Some(t.to_vec()),
            Self::List(l) => Some(l.borrow().clone()),
            _ => None,
        }
    }
}

/// Runs a pickle stream and returns whatever it left on the stack.
pub fn load(bytes: &[u8]) -> Result<Value, PtError> {
    Machine::new(bytes).run()
}

/// The virtual machine: a value stack, a memo, and a stack of mark positions.
struct Machine<'a> {
    bytes: &'a [u8],
    at: usize,
    stack: Vec<Value>,
    memo: Vec<Option<Value>>,
    marks: Vec<usize>,
}

impl<'a> Machine<'a> {
    /// Prepares a machine over a pickle stream.
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            stack: Vec::new(),
            memo: Vec::new(),
            marks: Vec::new(),
        }
    }

    /// Executes until `STOP`, returning the value it leaves behind.
    fn run(mut self) -> Result<Value, PtError> {
        loop {
            let at = self.at;
            let op = self.byte()?;
            match op {
                // -- framing, which carries no value ------------------------
                b'\x80' => {
                    self.byte()?;
                }
                b'\x95' => {
                    self.skip(8)?;
                }
                b'.' => {
                    return self.pop(at);
                }

                // -- atoms --------------------------------------------------
                b'N' => self.stack.push(Value::None),
                b'\x88' => self.stack.push(Value::Bool(true)),
                b'\x89' => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let v = self.byte()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'J' => {
                    let v = self.u32()? as i32;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                b'\x8a' | b'\x8b' => {
                    let len = if op == b'\x8a' {
                        usize::from(self.byte()?)
                    } else {
                        self.u32()? as usize
                    };
                    let body = self.take(len)?;
                    self.stack.push(Value::Int(long(body)));
                }
                b'G' => {
                    // Pickle floats are big-endian, unlike everything else here.
                    let body = self.take(8)?;
                    let bits = u64::from_be_bytes(body.try_into().expect("slice is 8 bytes"));
                    self.stack.push(Value::Float(f64::from_bits(bits)));
                }
                b'X' | b'\x8c' | b'\x8d' => {
                    let len = match op {
                        b'X' => self.u32()? as usize,
                        b'\x8c' => usize::from(self.byte()?),
                        _ => self.u64()? as usize,
                    };
                    let body = self.take(len)?;
                    let text = String::from_utf8_lossy(body).into_owned();
                    self.stack.push(Value::Str(Rc::from(text.as_str())));
                }
                b'B' | b'C' | b'\x8e' => {
                    let len = match op {
                        b'B' => self.u32()? as usize,
                        b'C' => usize::from(self.byte()?),
                        _ => self.u64()? as usize,
                    };
                    let body = self.take(len)?;
                    self.stack.push(Value::Bytes(Rc::from(body)));
                }

                // -- the memo -----------------------------------------------
                b'q' | b'r' => {
                    let slot = if op == b'q' {
                        usize::from(self.byte()?)
                    } else {
                        self.u32()? as usize
                    };
                    let top = self.peek(at)?;
                    self.put(slot, top);
                }
                b'\x94' => {
                    let top = self.peek(at)?;
                    let slot = self.memo.len();
                    self.put(slot, top);
                }
                b'h' | b'j' => {
                    let slot = if op == b'h' {
                        usize::from(self.byte()?)
                    } else {
                        self.u32()? as usize
                    };
                    let v = self
                        .memo
                        .get(slot)
                        .and_then(Option::as_ref)
                        .cloned()
                        .ok_or_else(|| PtError::PickleState {
                            at,
                            what: format!("memo slot {slot} has never been written"),
                        })?;
                    self.stack.push(v);
                }

                // -- containers ---------------------------------------------
                b'(' => self.marks.push(self.stack.len()),
                b')' => self.stack.push(Value::Tuple(Rc::from(Vec::new()))),
                b']' => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b'}' => self
                    .stack
                    .push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                b't' => {
                    let items = self.since_mark(at)?;
                    self.stack.push(Value::Tuple(Rc::from(items)));
                }
                b'\x85' | b'\x86' | b'\x87' => {
                    let n = usize::from(op - b'\x85') + 1;
                    let items = self.pop_n(at, n)?;
                    self.stack.push(Value::Tuple(Rc::from(items)));
                }
                b'a' => {
                    let v = self.pop(at)?;
                    self.list(at)?.borrow_mut().push(v);
                }
                b'e' => {
                    let items = self.since_mark(at)?;
                    self.list(at)?.borrow_mut().extend(items);
                }
                b's' => {
                    let [k, v]: [Value; 2] =
                        self.pop_n(at, 2)?.try_into().expect("pop_n returned 2");
                    self.dict(at)?.borrow_mut().push((k, v));
                }
                b'u' => {
                    let items = self.since_mark(at)?;
                    if !items.len().is_multiple_of(2) {
                        return Err(PtError::PickleState {
                            at,
                            what: "SETITEMS given an odd number of values".to_string(),
                        });
                    }
                    let dict = self.dict(at)?;
                    let mut dict = dict.borrow_mut();
                    for [k, v] in items.as_chunks::<2>().0 {
                        dict.push((k.clone(), v.clone()));
                    }
                }

                // -- imports, calls and persistent ids ------------------------
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global {
                        module: Rc::from(module.as_str()),
                        name: Rc::from(name.as_str()),
                    });
                }
                b'\x93' => {
                    let [module, name]: [Value; 2] =
                        self.pop_n(at, 2)?.try_into().expect("pop_n returned 2");
                    match (module, name) {
                        (Value::Str(module), Value::Str(name)) => {
                            self.stack.push(Value::Global { module, name });
                        }
                        (m, n) => {
                            return Err(PtError::PickleState {
                                at,
                                what: format!("STACK_GLOBAL given {} and {}", m.kind(), n.kind()),
                            });
                        }
                    }
                }
                b'R' | b'\x81' => {
                    let [callable, args]: [Value; 2] =
                        self.pop_n(at, 2)?.try_into().expect("pop_n returned 2");
                    let v = reduce(at, &callable, &args)?;
                    self.stack.push(v);
                }
                b'Q' => {
                    let pid = self.pop(at)?;
                    self.stack.push(persistent(at, &pid)?);
                }
                b'b' => {
                    // `obj.__setstate__(state)`. The one checkpoint this reader
                    // was written for passes `None`; a state dict never needs
                    // more, so a real state is dropped rather than applied, and
                    // the object is left as the stream built it.
                    let _state = self.pop(at)?;
                }

                _ => {
                    return Err(PtError::PickleOpcode {
                        opcode: op,
                        name: mnemonic(op),
                        at,
                    });
                }
            }
        }
    }

    /// Reads one byte.
    fn byte(&mut self) -> Result<u8, PtError> {
        let v = *self
            .bytes
            .get(self.at)
            .ok_or(PtError::PickleTruncated { at: self.at })?;
        self.at += 1;
        Ok(v)
    }

    /// Takes `n` bytes.
    fn take(&mut self, n: usize) -> Result<&'a [u8], PtError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or(PtError::PickleTruncated { at: self.at })?;
        let body = self
            .bytes
            .get(self.at..end)
            .ok_or(PtError::PickleTruncated { at: self.at })?;
        self.at = end;
        Ok(body)
    }

    /// Skips `n` bytes.
    fn skip(&mut self, n: usize) -> Result<(), PtError> {
        self.take(n).map(|_| ())
    }

    /// Reads a little-endian `u16`.
    fn u16(&mut self) -> Result<u16, PtError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes(b.try_into().expect("slice is 2 bytes")))
    }

    /// Reads a little-endian `u32`.
    fn u32(&mut self) -> Result<u32, PtError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().expect("slice is 4 bytes")))
    }

    /// Reads a little-endian `u64`.
    fn u64(&mut self) -> Result<u64, PtError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().expect("slice is 8 bytes")))
    }

    /// Reads a newline-terminated line, as protocol 0's `GLOBAL` uses.
    fn line(&mut self) -> Result<String, PtError> {
        let start = self.at;
        while self.byte()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.bytes[start..self.at - 1]).into_owned())
    }

    /// Writes a memo slot, growing the memo to reach it.
    fn put(&mut self, slot: usize, value: Value) {
        if slot >= self.memo.len() {
            self.memo.resize(slot + 1, None);
        }
        self.memo[slot] = Some(value);
    }

    /// Pops one value.
    fn pop(&mut self, at: usize) -> Result<Value, PtError> {
        self.stack.pop().ok_or_else(|| PtError::PickleState {
            at,
            what: "the stack is empty".to_string(),
        })
    }

    /// Clones the top value without popping it.
    fn peek(&self, at: usize) -> Result<Value, PtError> {
        self.stack
            .last()
            .cloned()
            .ok_or_else(|| PtError::PickleState {
                at,
                what: "the stack is empty".to_string(),
            })
    }

    /// Pops exactly `n` values, oldest first.
    fn pop_n(&mut self, at: usize, n: usize) -> Result<Vec<Value>, PtError> {
        if self.stack.len() < n {
            return Err(PtError::PickleState {
                at,
                what: format!("wanted {n} values and the stack holds {}", self.stack.len()),
            });
        }
        Ok(self.stack.split_off(self.stack.len() - n))
    }

    /// Pops everything pushed since the last `MARK`.
    fn since_mark(&mut self, at: usize) -> Result<Vec<Value>, PtError> {
        let mark = self.marks.pop().ok_or_else(|| PtError::PickleState {
            at,
            what: "no MARK is open".to_string(),
        })?;
        if mark > self.stack.len() {
            return Err(PtError::PickleState {
                at,
                what: "the stack shrank below its MARK".to_string(),
            });
        }
        Ok(self.stack.split_off(mark))
    }

    /// The list now on top of the stack, which `APPEND` mutates in place.
    fn list(&self, at: usize) -> Result<Rc<RefCell<Vec<Value>>>, PtError> {
        match self.stack.last() {
            Some(Value::List(l)) => Ok(Rc::clone(l)),
            other => Err(PtError::PickleState {
                at,
                what: format!(
                    "APPEND onto {}",
                    other.map_or("an empty stack", Value::kind)
                ),
            }),
        }
    }

    /// The dict now on top of the stack, which `SETITEM` mutates in place.
    fn dict(&self, at: usize) -> Result<Entries, PtError> {
        match self.stack.last() {
            Some(Value::Dict(d)) => Ok(Rc::clone(d)),
            other => Err(PtError::PickleState {
                at,
                what: format!(
                    "SETITEM onto {}",
                    other.map_or("an empty stack", Value::kind)
                ),
            }),
        }
    }
}

/// Calls one of the three names a state dict is allowed to contain.
fn reduce(at: usize, callable: &Value, args: &Value) -> Result<Value, PtError> {
    let Value::Global { module, name } = callable else {
        return Err(PtError::PickleState {
            at,
            what: format!("REDUCE on {}", callable.kind()),
        });
    };

    match (&**module, &**name) {
        // An empty ordered dict, which `SETITEMS` then fills.
        ("collections", "OrderedDict") => Ok(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
        // `(storage, storage_offset, size, stride, requires_grad, hooks, ..)`.
        // v3 appends a metadata argument and is otherwise identical, so both
        // read the same first four.
        ("torch._utils", "_rebuild_tensor_v2" | "_rebuild_tensor_v3") => rebuild_tensor(at, args),
        // A `Parameter` is its data plus a `requires_grad` flag that inference
        // has no use for.
        ("torch._utils", "_rebuild_parameter") => args
            .as_seq()
            .and_then(|a| a.first().cloned())
            .ok_or_else(|| PtError::PickleState {
                at,
                what: "_rebuild_parameter with no data argument".to_string(),
            }),
        _ => Err(PtError::UnsupportedGlobal {
            module: module.to_string(),
            name: name.to_string(),
        }),
    }
}

/// Rebuilds a tensor from `_rebuild_tensor_v2`'s arguments.
fn rebuild_tensor(at: usize, args: &Value) -> Result<Value, PtError> {
    let args = args.as_seq().ok_or_else(|| PtError::PickleState {
        at,
        what: format!("_rebuild_tensor_v2 given {}", args.kind()),
    })?;
    let bad = |what: &str| PtError::PickleState {
        at,
        what: format!("_rebuild_tensor_v2 {what}"),
    };
    if args.len() < 4 {
        return Err(bad("has fewer than four arguments"));
    }

    let Value::Storage(storage) = &args[0] else {
        return Err(bad("was not given a storage"));
    };
    let offset = args[1].as_usize().ok_or_else(|| bad("has a bad offset"))?;
    let dims = |v: &Value, what: &str| -> Result<Vec<usize>, PtError> {
        v.as_seq()
            .ok_or_else(|| bad(what))?
            .iter()
            .map(|d| d.as_usize().ok_or_else(|| bad(what)))
            .collect()
    };
    let shape = dims(&args[2], "has a bad shape")?;
    let stride = dims(&args[3], "has a bad stride")?;

    Ok(Value::Tensor(Rc::new(Tensor {
        storage: storage.clone(),
        offset,
        shape,
        stride,
    })))
}

/// Resolves a persistent id into a storage.
///
/// Torch writes `("storage", <storage class>, key, location, numel)`. The
/// location - `"cpu"`, `"cuda:0"` - says which device the tensor was saved
/// from and has no bearing on the bytes, so it is ignored.
fn persistent(at: usize, pid: &Value) -> Result<Value, PtError> {
    let parts = pid.as_seq().ok_or_else(|| PtError::PickleState {
        at,
        what: format!("a persistent id that is {}", pid.kind()),
    })?;
    let bad = |what: &str| PtError::PickleState {
        at,
        what: format!("a persistent id {what}"),
    };
    if parts.len() < 5 {
        return Err(bad("with fewer than five fields"));
    }
    match &parts[0] {
        Value::Str(tag) if &**tag == "storage" => {}
        _ => return Err(bad("that does not name a storage")),
    }

    let class = match &parts[1] {
        Value::Global { module, name } => format!("{module}.{name}"),
        // Newer torch writes the class name as a plain string.
        Value::Str(s) => s.to_string(),
        other => return Err(bad(&format!("whose type is {}", other.kind()))),
    };
    let key = match &parts[2] {
        Value::Str(s) => s.to_string(),
        Value::Int(v) => v.to_string(),
        other => return Err(bad(&format!("whose key is {}", other.kind()))),
    };
    let numel = parts[4]
        .as_usize()
        .ok_or_else(|| bad("with a bad length"))?;

    let dtype = Dtype::parse(&class).ok_or(PtError::UnsupportedStorage {
        key: key.clone(),
        class,
    })?;
    Ok(Value::Storage(Storage { key, dtype, numel }))
}

/// Decodes a `LONG1`/`LONG4` body: little-endian two's complement, any width.
fn long(body: &[u8]) -> i64 {
    if body.is_empty() {
        return 0;
    }
    let negative = body[body.len() - 1] & 0x80 != 0;
    let mut v: i64 = if negative { -1 } else { 0 };
    for (i, b) in body.iter().enumerate().take(8) {
        let mask = !(0xffi64 << (i * 8));
        v = (v & mask) | (i64::from(*b) << (i * 8));
    }
    v
}

/// The mnemonic for an opcode this reader does not implement.
///
/// Only the ones worth naming: an unimplemented opcode is a bug report, and
/// "0x8f (EMPTY_SET)" says what to add where "0x8f" does not.
fn mnemonic(op: u8) -> &'static str {
    match op {
        b'\x8f' => "EMPTY_SET",
        b'\x90' => "ADDITEMS",
        b'\x91' => "FROZENSET",
        b'\x92' => "NEWOBJ_EX",
        b'0' => "POP",
        b'2' => "DUP",
        b'I' => "INT",
        b'L' => "LONG",
        b'F' => "FLOAT",
        b'S' => "STRING",
        b'U' => "SHORT_BINSTRING",
        b'T' => "BINSTRING",
        b'V' => "UNICODE",
        b'g' => "GET",
        b'i' => "INST",
        b'l' => "LIST",
        b'd' => "DICT",
        b'o' => "OBJ",
        b'p' => "PUT",
        b'P' => "PERSID",
        _ => "unknown",
    }
}
