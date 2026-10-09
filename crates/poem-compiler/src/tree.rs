//! An op as a tree of its fields, with every `ValueId` marked, so the fusion
//! matcher and the tensor-parallel pass read, compare and renumber an op of
//! any family field by field without a hand-written walk per variant. It
//! holds only the kinds of field ops have: no strings, maps or `f64`s.

use poem_ir::{Operation, ValueId};
use serde::de::value::{MapDeserializer, SeqDeserializer};
use serde::de::{self, Deserialize, IntoDeserializer};
use serde::ser::{self, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    Id(u32),
    F32(u32),
    Int(i128),
    Bool(bool),
    Unit,
    None,
    Some(Box<Tree>),
    Seq(Vec<Tree>),
    Struct(Vec<(&'static str, Tree)>),
    Variant(&'static str, Box<Tree>),
}

#[must_use]
pub fn of(op: &Operation) -> Tree {
    op.serialize(Ser)
        .expect("an op serializes to a tree of its fields")
}

/// The op a tree was taken from.
#[must_use]
pub fn op(tree: Tree) -> Operation {
    Operation::deserialize(De(tree)).expect("a tree read back is the op it was taken from")
}

/// The fields of the struct an op's family and variant wrap; none for a
/// variant without fields.
#[must_use]
pub fn fields(tree: &Tree) -> &[(&'static str, Tree)] {
    match tree {
        Tree::Variant(_, inner) => fields(inner),
        Tree::Struct(fields) => fields,
        _ => &[],
    }
}

/// The field `name` of an op's struct.
#[must_use]
pub fn field<'t>(tree: &'t Tree, name: &str) -> Option<&'t Tree> {
    fields(tree)
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t)
}

/// The fields of the struct an op's family and variant wrap.
pub fn fields_mut(tree: &mut Tree) -> &mut Vec<(&'static str, Tree)> {
    match tree {
        Tree::Variant(_, inner) => fields_mut(inner),
        Tree::Struct(fields) => fields,
        other => panic!("{other:?} is not an op's fields"),
    }
}

/// The values a field holds, through an `Option` or a list.
#[must_use]
pub fn ids(tree: &Tree) -> Vec<ValueId> {
    match tree {
        Tree::Id(id) => vec![ValueId(*id)],
        Tree::Some(inner) => ids(inner),
        Tree::Seq(items) => items.iter().flat_map(ids).collect(),
        _ => Vec::new(),
    }
}

/// `op` with every value renumbered by `f`.
#[must_use]
pub fn remap(op: &Operation, f: &impl Fn(ValueId) -> ValueId) -> Operation {
    Operation::deserialize(De(renumbered(of(op), f)))
        .expect("a renumbered op reads back as the op it was")
}

/// `tree` with every value renumbered by `f`.
#[must_use]
pub fn renumbered(tree: Tree, f: &impl Fn(ValueId) -> ValueId) -> Tree {
    match tree {
        Tree::Id(id) => Tree::Id(f(ValueId(id)).0),
        Tree::Some(inner) => Tree::Some(Box::new(renumbered(*inner, f))),
        Tree::Seq(items) => Tree::Seq(items.into_iter().map(|t| renumbered(t, f)).collect()),
        Tree::Struct(fields) => Tree::Struct(
            fields
                .into_iter()
                .map(|(k, v)| (k, renumbered(v, f)))
                .collect(),
        ),
        Tree::Variant(name, inner) => Tree::Variant(name, Box::new(renumbered(*inner, f))),
        leaf => leaf,
    }
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl ser::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}

impl de::Error for Error {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}

struct Ser;

fn unheld(kind: &str) -> Error {
    Error(format!("an op field is {kind}, which a tree does not hold"))
}

pub struct Items(Vec<Tree>, Option<&'static str>);
pub struct Fields(Vec<(&'static str, Tree)>, Option<&'static str>);

fn wrap(variant: Option<&'static str>, tree: Tree) -> Tree {
    match variant {
        Some(name) => Tree::Variant(name, Box::new(tree)),
        None => tree,
    }
}

impl ser::Serializer for Ser {
    type Ok = Tree;
    type Error = Error;
    type SerializeSeq = Items;
    type SerializeTuple = Items;
    type SerializeTupleStruct = Items;
    type SerializeTupleVariant = Items;
    type SerializeMap = ser::Impossible<Tree, Error>;
    type SerializeStruct = Fields;
    type SerializeStructVariant = Fields;

    fn serialize_bool(self, v: bool) -> Result<Tree, Error> {
        Ok(Tree::Bool(v))
    }
    fn serialize_i8(self, v: i8) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_i16(self, v: i16) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_i32(self, v: i32) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_i64(self, v: i64) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_u8(self, v: u8) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_u16(self, v: u16) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_u32(self, v: u32) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_u64(self, v: u64) -> Result<Tree, Error> {
        Ok(Tree::Int(v.into()))
    }
    fn serialize_f32(self, v: f32) -> Result<Tree, Error> {
        Ok(Tree::F32(v.to_bits()))
    }
    fn serialize_f64(self, _: f64) -> Result<Tree, Error> {
        Err(unheld("an f64"))
    }
    fn serialize_char(self, _: char) -> Result<Tree, Error> {
        Err(unheld("a char"))
    }
    fn serialize_str(self, _: &str) -> Result<Tree, Error> {
        Err(unheld("a string"))
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<Tree, Error> {
        Err(unheld("bytes"))
    }
    fn serialize_none(self) -> Result<Tree, Error> {
        Ok(Tree::None)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Tree, Error> {
        Ok(Tree::Some(Box::new(value.serialize(Ser)?)))
    }
    fn serialize_unit(self) -> Result<Tree, Error> {
        Ok(Tree::Unit)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Tree, Error> {
        Ok(Tree::Unit)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Tree, Error> {
        Ok(Tree::Variant(variant, Box::new(Tree::Unit)))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Tree, Error> {
        let inner = value.serialize(Ser)?;
        match (name, inner) {
            ("ValueId", Tree::Int(id)) => Ok(Tree::Id(
                u32::try_from(id).map_err(|_| Error("a value id fits 32 bits".into()))?,
            )),
            (_, inner) => Ok(inner),
        }
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Tree, Error> {
        Ok(Tree::Variant(variant, Box::new(value.serialize(Ser)?)))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Items, Error> {
        Ok(Items(Vec::with_capacity(len.unwrap_or(0)), None))
    }
    fn serialize_tuple(self, len: usize) -> Result<Items, Error> {
        Ok(Items(Vec::with_capacity(len), None))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Items, Error> {
        Ok(Items(Vec::with_capacity(len), None))
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Items, Error> {
        Ok(Items(Vec::with_capacity(len), Some(variant)))
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self::SerializeMap, Error> {
        Err(unheld("a map"))
    }
    fn serialize_struct(self, _: &'static str, len: usize) -> Result<Fields, Error> {
        Ok(Fields(Vec::with_capacity(len), None))
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Fields, Error> {
        Ok(Fields(Vec::with_capacity(len), Some(variant)))
    }
}

macro_rules! items {
    ($($trait:ident :: $method:ident),+) => {
        $(
            impl ser::$trait for Items {
                type Ok = Tree;
                type Error = Error;
                fn $method<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
                    self.0.push(value.serialize(Ser)?);
                    Ok(())
                }
                fn end(self) -> Result<Tree, Error> {
                    Ok(wrap(self.1, Tree::Seq(self.0)))
                }
            }
        )+
    };
}

items!(
    SerializeSeq::serialize_element,
    SerializeTuple::serialize_element,
    SerializeTupleStruct::serialize_field,
    SerializeTupleVariant::serialize_field
);

macro_rules! fields {
    ($($trait:ident),+) => {
        $(
            impl ser::$trait for Fields {
                type Ok = Tree;
                type Error = Error;
                fn serialize_field<T: ?Sized + Serialize>(
                    &mut self,
                    key: &'static str,
                    value: &T,
                ) -> Result<(), Error> {
                    self.0.push((key, value.serialize(Ser)?));
                    Ok(())
                }
                fn end(self) -> Result<Tree, Error> {
                    Ok(wrap(self.1, Tree::Struct(self.0)))
                }
            }
        )+
    };
}

fields!(SerializeStruct, SerializeStructVariant);

pub struct De(Tree);

impl<'de> IntoDeserializer<'de, Error> for Tree {
    type Deserializer = De;
    fn into_deserializer(self) -> De {
        De(self)
    }
}

impl<'de> de::Deserializer<'de> for De {
    type Error = Error;

    fn deserialize_any<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self.0 {
            Tree::Id(id) => visitor.visit_u32(id),
            Tree::F32(bits) => visitor.visit_f32(f32::from_bits(bits)),
            Tree::Int(n) => match u64::try_from(n) {
                Ok(n) => visitor.visit_u64(n),
                Err(_) => visitor.visit_i64(
                    i64::try_from(n).map_err(|_| Error("an integer field fits 64 bits".into()))?,
                ),
            },
            Tree::Bool(b) => visitor.visit_bool(b),
            Tree::Unit => visitor.visit_unit(),
            Tree::None => visitor.visit_none(),
            Tree::Some(inner) => visitor.visit_some(De(*inner)),
            Tree::Seq(items) => visitor.visit_seq(SeqDeserializer::new(items.into_iter())),
            Tree::Struct(fields) => visitor.visit_map(MapDeserializer::new(fields.into_iter())),
            Tree::Variant(..) => Err(Error("a variant read where no enum is".into())),
        }
    }

    fn deserialize_option<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self.0 {
            Tree::None | Tree::Unit => visitor.visit_none(),
            Tree::Some(inner) => visitor.visit_some(De(*inner)),
            other => visitor.visit_some(De(other)),
        }
    }

    fn deserialize_newtype_struct<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: de::Visitor<'de>>(
        self,
        _: &'static str,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match self.0 {
            Tree::Variant(name, inner) => visitor.visit_enum(Variant(name, *inner)),
            other => Err(Error(format!("an enum read from {other:?}"))),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct seq tuple tuple_struct map struct
        identifier ignored_any
    }
}

struct Variant(&'static str, Tree);

impl<'de> de::EnumAccess<'de> for Variant {
    type Error = Error;
    type Variant = De;

    fn variant_seed<S: de::DeserializeSeed<'de>>(self, seed: S) -> Result<(S::Value, De), Error> {
        let name = seed.deserialize(self.0.into_deserializer())?;
        Ok((name, De(self.1)))
    }
}

impl<'de> de::VariantAccess<'de> for De {
    type Error = Error;

    fn unit_variant(self) -> Result<(), Error> {
        Ok(())
    }

    fn newtype_variant_seed<S: de::DeserializeSeed<'de>>(self, seed: S) -> Result<S::Value, Error> {
        seed.deserialize(self)
    }

    fn tuple_variant<V: de::Visitor<'de>>(self, _: usize, visitor: V) -> Result<V::Value, Error> {
        de::Deserializer::deserialize_any(self, visitor)
    }

    fn struct_variant<V: de::Visitor<'de>>(
        self,
        _: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        de::Deserializer::deserialize_any(self, visitor)
    }
}
