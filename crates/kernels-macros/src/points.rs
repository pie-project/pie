use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{
    Attribute, BinOp, Block, Error, Expr, ExprAssign, FnArg, GenericParam, Ident, ItemTrait, Lit,
    Member, Pat, PathSegment, ReceiverKind, ReturnType, Signature, Token, TraitItem, Type,
    TypeParamBound, WherePredicate, parse_quote, spanned::Spanned,
};

pub fn expand(mut item: ItemTrait) -> Result<TokenStream, Error> {
    let family = snake(&item.ident);
    let table = Ident::new(
        &format!("{}_POINTS", family.to_uppercase()),
        item.ident.span(),
    );
    let points = item
        .items
        .iter()
        .filter_map(|i| match i {
            TraitItem::Fn(f) => Some(f),
            _ => None,
        })
        .map(|f| point(format!("{family}.{}", f.sig.ident), &f.attrs, &f.sig))
        .collect::<Result<Vec<_>, _>>()?;
    for i in &mut item.items {
        if let TraitItem::Fn(f) = i {
            strip_shape(&mut f.attrs);
            if f.default.is_none() {
                f.default = Some(unclaimed(&format!("{family}.{}", f.sig.ident), &f.sig));
                f.semi_token = None;
            }
        }
    }
    Ok(quote! {
        #item

        pub const #table: &[Point] = &[#(#points),*];
    })
}

fn unclaimed(name: &str, sig: &Signature) -> Block {
    let slots = sig.inputs.iter().filter_map(|a| match a {
        FnArg::Typed(pt) => match &*pt.pat {
            Pat::Ident(id) => Some(id.ident.clone()),
            _ => None,
        },
        FnArg::Receiver(_) => None,
    });
    parse_quote!({
        let _ = (#(#slots,)*);
        Err(Refusal::unclaimed(#name))
    })
}

pub(crate) fn strip_shape(attrs: &mut Vec<Attribute>) {
    attrs.retain(|a| !a.path().is_ident("shape"));
}

pub(crate) fn snake(ident: &Ident) -> String {
    let spelled: Vec<char> = ident.to_string().chars().collect();
    let mut out = String::with_capacity(spelled.len() + 4);
    for (i, c) in spelled.iter().enumerate() {
        let breaks = i > 0
            && c.is_uppercase()
            && (!spelled[i - 1].is_uppercase()
                || spelled.get(i + 1).is_some_and(char::is_ascii_lowercase));
        if breaks {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}

pub(crate) fn point(
    name: String,
    attrs: &[Attribute],
    sig: &Signature,
) -> Result<TokenStream, Error> {
    let Axes { scalars, reprs } = axes(&name, sig)?;
    answers(&name, sig)?;
    let mut args = sig.inputs.iter();
    match args.next() {
        Some(FnArg::Receiver(r)) if matches!(r.kind, ReceiverKind::Reference(_, _, None)) => {}
        _ => {
            return Err(Error::new(
                sig.span(),
                format!("`{name}` takes `&self` first: the plane the point fires on"),
            ));
        }
    }
    let decls = read(&name, args, &scalars, &reprs)?;
    let slots = decls.iter().map(Decl::row);
    let outs = outs(&name, attrs, &decls)?;
    let n = scalars.len();
    let r = reprs.len();
    Ok(
        quote!(Point { name: #name, axes: #n, reprs: #r, slots: &[#(#slots),*], outs: &[#(#outs),*] }),
    )
}

struct Decl {
    name: Ident,
    kind: Kind,

    dtype: TokenStream,

    axis: Option<usize>,

    prim: Option<Ident>,

    bank: bool,

    column: usize,

    countable: bool,
    span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    In,
    InOut,
    Out,
    Const,
    Cache,
    Scalar,
}

impl Decl {
    fn row(&self) -> TokenStream {
        let name = self.name.to_string();
        let mark = Ident::new(
            match self.kind {
                Kind::In => "In",
                Kind::InOut => "InOut",
                Kind::Out => "Out",
                Kind::Const => "Const",
                Kind::Cache => "Cache",
                Kind::Scalar => "Scalar",
            },
            self.span,
        );
        let dtype = &self.dtype;
        quote!(Slot { name: #name, mark: Mark::#mark, dtype: #dtype })
    }

    fn operand(&self) -> bool {
        matches!(self.kind, Kind::In | Kind::InOut)
    }
}

fn read<'a>(
    point: &str,
    args: impl Iterator<Item = &'a FnArg>,
    axes: &[Ident],
    reprs: &[Ident],
) -> Result<Vec<Decl>, Error> {
    let (mut operands, mut weights, mut params) = (0, 0, 0);

    let mut banked = false;

    let mut minted: Option<String> = None;
    let mut pooled: Option<String> = None;
    let mut out = Vec::new();
    for arg in args {
        let mut d = slot(point, arg, axes, reprs)?;
        d.countable = true;
        if let Some(result) = &minted
            && d.kind != Kind::Out
        {
            return Err(Error::new(
                d.span,
                format!(
                    "`{point}`: slot `{}` stands after the result `{result}` — a point's \
                     `Out` slots come last. A statement's operands are this list with the \
                     results taken off the end, so a result in the middle renumbers every \
                     slot behind it in the generated builder and refuses nowhere",
                    d.name
                ),
            ));
        }
        if d.kind == Kind::Out {
            minted.get_or_insert_with(|| d.name.to_string());
        }
        if d.kind == Kind::Cache {
            if let Some(first) = &pooled {
                return Err(Error::new(
                    d.span,
                    format!(
                        "`{point}`: slot `{}` is a second `Cache` beside `{first}` — a \
                         statement names ONE pool row, which is why `BoundOp::recurrent` \
                         and `BoundOp::pages` take no index. Both slots would bind that \
                         one row",
                        d.name
                    ),
                ));
            }
            pooled = Some(d.name.to_string());
        }
        match d.kind {
            Kind::In | Kind::InOut => {
                d.column = operands;
                operands += 1;
            }
            Kind::Const => {
                d.column = weights;
                d.countable = !banked;
                banked |= d.bank;
                weights += 1;
            }
            Kind::Scalar => {
                d.column = params;
                params += 1;
            }

            Kind::Out | Kind::Cache => {}
        }
        out.push(d);
    }
    Ok(out)
}

fn outs(point: &str, attrs: &[Attribute], decls: &[Decl]) -> Result<Vec<TokenStream>, Error> {
    let results: Vec<&Decl> = decls
        .iter()
        .filter(|d| matches!(d.kind, Kind::Out | Kind::InOut))
        .collect();
    let stated = attrs.iter().find(|a| a.path().is_ident("shape"));
    if let Some(extra) = attrs.iter().filter(|a| a.path().is_ident("shape")).nth(1) {
        return Err(Error::new(
            extra.span(),
            format!("`{point}` states its results' sizes once, in one `#[shape]`"),
        ));
    }

    let mints = results.iter().any(|d| d.kind == Kind::Out);
    let rides = results.iter().any(|d| d.kind == Kind::InOut);
    if mints && rides {
        return Err(Error::new(
            results[0].span,
            format!(
                "`{point}` mints an `Out` and writes through an `InOut`: a statement's results \
                 are one or the other, because a rule can size what it mints and can only \
                 restate what it was handed"
            ),
        ));
    }
    if !mints {
        if let Some(a) = stated {
            return Err(Error::new(
                a.span(),
                format!(
                    "`{point}` mints nothing: an `InOut` result is the rectangle its operand \
                     already is, and an effect states no rectangle at all"
                ),
            ));
        }
        return Ok(results
            .iter()
            .map(|d| {
                let at = d.column;
                quote!(Shape {
                    rows: Fan::Ride(#at),
                    width: Width::Of(#at),
                    elem: Element::Ride(#at)
                })
            })
            .collect());
    }

    let Some(attr) = stated else {
        let named: Vec<String> = results.iter().map(|d| d.name.to_string()).collect();
        return Err(Error::new(
            results[0].span,
            format!(
                "`{point}` mints {} and states no `#[shape]`: how wide a result is is a fact \
                 of this declaration, so write it here — `#[shape({} = ..)]`",
                named.join(", "),
                named.join(" = .., ")
            ),
        ));
    };
    let clauses = attr.parse_args_with(Punctuated::<ExprAssign, Token![,]>::parse_terminated)?;

    let mut rows = Vec::new();
    for d in &results {
        let mut found = None;
        for c in &clauses {
            let Expr::Path(p) = &*c.left else {
                return Err(Error::new(
                    c.left.span(),
                    format!("`{point}`: a `#[shape]` clause names a result — `y = ..`"),
                ));
            };
            if p.path.is_ident(&d.name) {
                if found.is_some() {
                    return Err(Error::new(
                        c.span(),
                        format!("`{point}`: result `{}` is sized twice", d.name),
                    ));
                }
                found = Some(&*c.right);
            }
        }
        let Some(expr) = found else {
            return Err(Error::new(
                attr.span(),
                format!(
                    "`{point}`: result `{}` has no size — every `Out` slot states one",
                    d.name
                ),
            ));
        };
        rows.push(shape(point, decls, d, expr)?);
    }

    for c in &clauses {
        let Expr::Path(p) = &*c.left else { continue };
        if !results.iter().any(|d| p.path.is_ident(&d.name)) {
            return Err(Error::new(
                c.left.span(),
                format!(
                    "`{point}`: `{}` is not a result of this point",
                    quoted(&c.left)
                ),
            ));
        }
    }
    Ok(rows)
}

fn shape(point: &str, decls: &[Decl], out: &Decl, expr: &Expr) -> Result<TokenStream, Error> {
    let elem = element(point, decls, out)?;

    if let Expr::Path(_) = expr {
        let at = operand_of(point, decls, expr)?.column;
        return Ok(quote!(Shape { rows: Fan::Ride(#at), width: Width::Of(#at), elem: #elem }));
    }
    let Expr::Array(a) = expr else {
        return Err(Error::new(
            expr.span(),
            format!(
                "`{point}`: a result is an operand's own rectangle (`y = x`) or a rows-and-width \
                 pair (`y = [x.rows, intermediate]`)"
            ),
        ));
    };
    let [r, w] = a.elems.iter().collect::<Vec<_>>()[..] else {
        return Err(Error::new(
            a.span(),
            format!("`{point}`: a stated rectangle is `[rows, width]` and nothing longer"),
        ));
    };
    let rows = fan(point, decls, r)?;
    let width = width(point, decls, w)?;
    Ok(quote!(Shape { rows: #rows, width: #width, elem: #elem }))
}

fn fan(point: &str, decls: &[Decl], expr: &Expr) -> Result<TokenStream, Error> {
    match expr {
        Expr::Path(p) if p.path.is_ident("fire") => Ok(quote!(Fan::Fire)),

        Expr::Field(f) if member_is(&f.member, "rows") => {
            let d = operand_of(point, decls, &f.base)?;
            let at = d.column;
            Ok(quote!(Fan::Ride(#at)))
        }

        Expr::Call(c) if matches!(&*c.func, Expr::Path(p) if p.path.is_ident("per")) => {
            let [arg] = c.args.iter().collect::<Vec<_>>()[..] else {
                return Err(Error::new(
                    c.span(),
                    format!("`{point}`: `per(..)` names one operand — the fan it counts"),
                ));
            };
            let d = operand_of(point, decls, arg)?;
            let at = d.column;
            Ok(quote!(Fan::Per(#at)))
        }
        _ => Err(Error::new(
            expr.span(),
            format!(
                "`{point}`: a row count is `fire`, an operand's `x.rows`, or `per(x)` — one row \
                 per element of that operand's width"
            ),
        )),
    }
}

fn width(point: &str, decls: &[Decl], expr: &Expr) -> Result<TokenStream, Error> {
    match expr {
        Expr::Paren(p) => width(point, decls, &p.expr),
        Expr::Group(g) => width(point, decls, &g.expr),

        Expr::Lit(l) => match &l.lit {
            Lit::Int(n) => {
                let n: u64 = n.base10_parse()?;
                Ok(quote!(Width::Count(#n)))
            }
            other => Err(Error::new(
                other.span(),
                format!("`{point}`: a width is a whole number of elements"),
            )),
        },
        Expr::Binary(b) => {
            let left = width(point, decls, &b.left)?;
            let right = width(point, decls, &b.right)?;
            let op = match b.op {
                BinOp::Mul(_) => quote!(Times),
                BinOp::Div(_) => quote!(Over),
                BinOp::Sub(_) => quote!(Less),
                _ => {
                    return Err(Error::new(
                        b.op.span(),
                        format!(
                            "`{point}`: a width multiplies, divides exactly, or subtracts — \
                             every rule this floor states is one of the three"
                        ),
                    ));
                }
            };
            Ok(quote!(Width::#op(&#left, &#right)))
        }

        Expr::Field(f) if member_is(&f.member, "width") => {
            let d = operand_of(point, decls, &f.base)?;
            let at = d.column;
            Ok(quote!(Width::Of(#at)))
        }

        Expr::MethodCall(m) if m.method == "axis" => {
            let d = slot_named(point, decls, &m.receiver)?;
            if d.kind != Kind::Const {
                return Err(Error::new(
                    m.receiver.span(),
                    format!(
                        "`{point}`: `{}` is not a `Const` — an axis is a dimension of a WEIGHT, \
                         read off the parameter the Load contract registered",
                        d.name
                    ),
                ));
            }
            if !d.countable {
                return Err(Error::new(
                    m.receiver.span(),
                    format!(
                        "`{point}`: `{}` stands after a quantised bank, so its weight column is \
                         `R::PLANES` wide and no number this macro can count",
                        d.name
                    ),
                ));
            }
            let [arg] = m.args.iter().collect::<Vec<_>>()[..] else {
                return Err(Error::new(
                    m.span(),
                    format!("`{point}`: `.axis(n)` names one dimension"),
                ));
            };
            let Expr::Lit(l) = arg else {
                return Err(Error::new(
                    arg.span(),
                    format!("`{point}`: `.axis(n)` takes a literal dimension"),
                ));
            };
            let Lit::Int(n) = &l.lit else {
                return Err(Error::new(
                    l.span(),
                    format!("`{point}`: `.axis(n)` takes a literal dimension"),
                ));
            };
            let dim: usize = n.base10_parse()?;
            let at = d.column;
            Ok(quote!(Width::Axis(#at, #dim)))
        }

        Expr::Path(_) => {
            let d = slot_named(point, decls, expr)?;
            if d.kind != Kind::Scalar {
                return Err(Error::new(
                    expr.span(),
                    format!(
                        "`{point}`: `{}` is a rectangle and not a number — write `{}.width` for \
                         how wide it is",
                        d.name, d.name
                    ),
                ));
            }

            match d.prim.as_ref().map(Ident::to_string).as_deref() {
                Some("I32" | "U32") => {}
                other => {
                    return Err(Error::new(
                        expr.span(),
                        format!(
                            "`{point}`: `{}` is a `{}` and a width is a COUNT — the statement's \
                             params column carries an `f32` as its bit pattern, so a width read \
                             off one is a rectangle of a billion elements and no refusal \
                             anywhere",
                            d.name,
                            other.unwrap_or("?").to_lowercase(),
                        ),
                    ));
                }
            }
            let at = d.column;
            Ok(quote!(Width::Stated(#at)))
        }
        _ => Err(Error::new(
            expr.span(),
            format!(
                "`{point}`: a width is an operand's `x.width`, a stated scalar, a weight's \
                 `w.axis(n)`, a number, or those multiplied, divided and subtracted"
            ),
        )),
    }
}

fn element(point: &str, decls: &[Decl], out: &Decl) -> Result<TokenStream, Error> {
    if let Some(p) = &out.prim {
        if p == "Bool" {
            return Err(Error::new(
                out.span,
                format!(
                    "`{point}`: result `{}` is a rectangle of `bool`, and `bool` is a host \
                     scalar's run on this floor — no arena mints one and no walk can size it",
                    out.name
                ),
            ));
        }
        return Ok(quote!(Element::Fixed(Prim::#p)));
    }
    let Some(axis) = out.axis else {
        return Err(Error::new(
            out.span,
            format!(
                "`{point}`: result `{}` carries no element — a result is a rectangle of \
                 elements, so it is `Out<Self::Tensor<..>>` and not a bank or a pool view",
                out.name
            ),
        ));
    };
    let on_axis = |want: fn(&Decl) -> bool| decls.iter().find(|d| want(d) && d.axis == Some(axis));
    if let Some(d) = on_axis(Decl::operand) {
        let at = d.column;
        return Ok(quote!(Element::Ride(#at)));
    }
    if let Some(d) = on_axis(|d| d.kind == Kind::Const && d.countable) {
        let at = d.column;
        return Ok(quote!(Element::Weight(#at)));
    }
    Ok(quote!(Element::Activation))
}

fn member_is(m: &Member, named: &str) -> bool {
    matches!(m, Member::Named(id) if id == named)
}

fn slot_named<'d>(point: &str, decls: &'d [Decl], expr: &Expr) -> Result<&'d Decl, Error> {
    let Expr::Path(p) = expr else {
        return Err(Error::new(
            expr.span(),
            format!("`{point}`: a `#[shape]` names slots of this method, one word each"),
        ));
    };
    decls
        .iter()
        .find(|d| p.path.is_ident(&d.name))
        .ok_or_else(|| {
            Error::new(
                expr.span(),
                format!("`{point}`: `{}` is not a slot of this method", quoted(expr)),
            )
        })
}

fn operand_of<'d>(point: &str, decls: &'d [Decl], expr: &Expr) -> Result<&'d Decl, Error> {
    let d = slot_named(point, decls, expr)?;
    if !d.operand() {
        return Err(Error::new(
            expr.span(),
            format!(
                "`{point}`: `{}` is not an operand — only an `In` or `InOut` slot carries a \
                 rectangle a rule can read rows and a width off",
                d.name
            ),
        ));
    }
    Ok(d)
}

fn quoted(expr: &Expr) -> String {
    match expr {
        Expr::Path(p) => p
            .path
            .get_ident()
            .map_or_else(|| "that".to_string(), std::string::ToString::to_string),
        _ => "that".to_string(),
    }
}

struct Axes {
    scalars: Vec<Ident>,

    reprs: Vec<Ident>,
}

fn axes(point: &str, sig: &Signature) -> Result<Axes, Error> {
    let mut out = Axes {
        scalars: Vec::new(),
        reprs: Vec::new(),
    };
    for p in &sig.generics.params {
        let GenericParam::Type(t) = p else {
            return Err(Error::new(
                p.span(),
                format!(
                    "`{point}`: a point's generics are its axes and nothing else \
                     — write `<T: Scalar>` for an element, `<R: Repr>` for a bank"
                ),
            ));
        };
        let elsewhere: Vec<&TypeParamBound> = sig
            .generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .filter_map(|p| match p {
                WherePredicate::Type(w) => match &w.bounded_ty {
                    Type::Path(path) if path.path.is_ident(&t.ident) => Some(&w.bounds),
                    _ => None,
                },
                _ => None,
            })
            .flatten()
            .collect();
        let bounds = || t.bounds.iter().chain(elsewhere.iter().copied());
        if bounds().any(|b| bound_is(b, "Scalar")) {
            if !out.reprs.is_empty() {
                return Err(Error::new(
                    t.span(),
                    format!(
                        "`{point}`: axis `{}` is a `Scalar` behind a `Repr` — a point's \
                         element axes come first, so a turbofish is `axes` then `reprs`",
                        t.ident
                    ),
                ));
            }
            out.scalars.push(t.ident.clone());
        } else if bounds().any(|b| bound_is(b, "Repr")) {
            out.reprs.push(t.ident.clone());
        } else {
            return Err(Error::new(
                t.span(),
                format!(
                    "`{point}`: axis `{}` states no bound — a point quantifies over \
                     `Scalar` (an element) or `Repr` (a bank's storage form)",
                    t.ident
                ),
            ));
        }
    }
    Ok(out)
}

fn bound_is(b: &TypeParamBound, what: &str) -> bool {
    matches!(b, TypeParamBound::Trait(t) if t.path.segments.last().is_some_and(|s| s.ident == what))
}

fn answers(point: &str, sig: &Signature) -> Result<(), Error> {
    let stated = match &sig.output {
        ReturnType::Type(_, ty) => match &**ty {
            Type::Path(p) => p.path.segments.last().is_some_and(|s| {
                s.ident == "Result"
                    && match args(s).as_slice() {
                        [Type::Tuple(unit), Type::Path(r)] => {
                            unit.elems.is_empty()
                                && r.path.segments.last().is_some_and(|s| s.ident == "Refusal")
                        }
                        _ => false,
                    }
            }),
            _ => false,
        },
        ReturnType::Default => false,
    };
    if stated {
        return Ok(());
    }
    Err(Error::new(
        sig.output.span(),
        format!("`{point}` answers `Result<(), Refusal>`, and a point answers nothing else"),
    ))
}

fn slot(point: &str, arg: &FnArg, axes: &[Ident], reprs: &[Ident]) -> Result<Decl, Error> {
    let FnArg::Typed(pt) = arg else {
        return Err(Error::new(
            arg.span(),
            format!("`{point}`: `&self` is the first parameter and occurs once"),
        ));
    };
    let Pat::Ident(id) = &*pt.pat else {
        return Err(Error::new(
            pt.pat.span(),
            format!("`{point}`: a slot is named, and this one is a pattern"),
        ));
    };
    let name = id.ident.clone();
    let (kind, rides) = marked(point, &name.to_string(), &pt.ty, axes, reprs)?;
    Ok(Decl {
        name,
        kind,
        dtype: rides.dtype,
        axis: rides.axis,
        prim: rides.prim,
        bank: rides.bank,
        column: 0,
        countable: true,
        span: pt.span(),
    })
}

struct Rides {
    dtype: TokenStream,
    axis: Option<usize>,
    prim: Option<Ident>,
    bank: bool,
}

impl Rides {
    fn generic(at: usize) -> Rides {
        Rides {
            dtype: quote!(Dtype::Generic(#at)),
            axis: Some(at),
            prim: None,
            bank: false,
        }
    }

    fn fixed(p: Ident) -> Rides {
        Rides {
            dtype: quote!(Dtype::Fixed(Prim::#p)),
            axis: None,
            prim: Some(p),
            bank: false,
        }
    }
}

fn marked(
    point: &str,
    slot: &str,
    ty: &Type,
    axes: &[Ident],
    reprs: &[Ident],
) -> Result<(Kind, Rides), Error> {
    if let Some(p) = prim(ty) {
        return Ok((Kind::Scalar, Rides::fixed(p)));
    }
    let marked = match ty {
        Type::Path(p) => p.path.segments.last().filter(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "In" | "InOut" | "Out" | "Const" | "Cache"
            )
        }),
        _ => None,
    };
    let Some(seg) = marked else {
        return Err(Error::new(
            ty.span(),
            format!(
                "`{point}`: slot `{slot}` is neither marked nor a host scalar — an operand is \
                 `In`, `InOut`, `Out` or `Const` of `Self::Tensor<..>`, a pool row is `Cache` \
                 of the plane's view, a scalar is a bare `f32`, `i32`, `u32` or `bool`"
            ),
        ));
    };
    let carried = args(seg);
    let [payload] = carried.as_slice() else {
        return Err(Error::new(
            seg.span(),
            format!(
                "`{point}`: mark `{}` on slot `{slot}` carries one payload",
                seg.ident
            ),
        ));
    };

    let rides = if seg.ident == "Cache" {
        pool(point, slot, payload)?
    } else if let Some(bank) = bank(point, slot, payload, reprs) {
        bank?
    } else {
        tensor(point, slot, payload, axes)?
    };
    let kind = match seg.ident.to_string().as_str() {
        "In" => Kind::In,
        "InOut" => Kind::InOut,
        "Out" => Kind::Out,
        "Const" => Kind::Const,
        _ => Kind::Cache,
    };
    Ok((kind, rides))
}

fn payload<'t>(ty: &'t Type, named: &str) -> Option<&'t PathSegment> {
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last().filter(|s| s.ident == named)?;
    match &p.qself {
        Some(q) => matches!(&*q.ty, Type::Path(s)
            if s.qself.is_none() && s.path.is_ident("Self"))
        .then_some(seg),
        None => (p.path.segments.len() == 2 && p.path.segments[0].ident == "Self").then_some(seg),
    }
}

fn bank(point: &str, slot: &str, ty: &Type, reprs: &[Ident]) -> Option<Result<Rides, Error>> {
    let seg = payload(ty, "Bank")?;
    let carried = args(seg);
    let [repr] = carried.as_slice() else {
        return Some(Err(Error::new(
            seg.span(),
            format!("`{point}`: `Self::Bank` on slot `{slot}` carries one repr"),
        )));
    };
    if let Type::Path(p) = repr
        && let Some(i) = p
            .path
            .get_ident()
            .and_then(|id| reprs.iter().position(|a| a == id))
    {
        return Some(Ok(Rides {
            dtype: quote!(Dtype::Bank(#i)),
            axis: None,
            prim: None,
            bank: true,
        }));
    }
    Some(Err(Error::new(
        repr.span(),
        format!(
            "`{point}`: slot `{slot}` carries a bank at a repr that is not an axis of this \
             method — write `Self::Bank<R>` for `<R: Repr>`"
        ),
    )))
}

fn pool(point: &str, slot: &str, ty: &Type) -> Result<Rides, Error> {
    let named = payload(ty, "Recurrent")
        .or_else(|| payload(ty, "Pages"))
        .filter(|s| s.arguments.is_none());
    if named.is_some() {
        return Ok(Rides {
            dtype: quote!(Dtype::Opaque),
            axis: None,
            prim: None,
            bank: false,
        });
    }
    Err(Error::new(
        ty.span(),
        format!(
            "`{point}`: `Cache` on slot `{slot}` carries the pool's own view, and the \
             floor declares two — `Self::Recurrent` and `Self::Pages`"
        ),
    ))
}

fn tensor(point: &str, slot: &str, ty: &Type, axes: &[Ident]) -> Result<Rides, Error> {
    let Some(seg) = payload(ty, "Tensor") else {
        return Err(Error::new(
            ty.span(),
            format!(
                "`{point}`: a mark on slot `{slot}` carries the plane's payload, \
                 `Self::Tensor<..>` or `Self::Bank<..>` — a host scalar is bare and \
                 wears no mark"
            ),
        ));
    };
    let carried = args(seg);
    let [dtype] = carried.as_slice() else {
        return Err(Error::new(
            seg.span(),
            format!("`{point}`: `Self::Tensor` on slot `{slot}` carries one dtype"),
        ));
    };
    if let Type::Path(p) = dtype
        && let Some(i) = p
            .path
            .get_ident()
            .and_then(|id| axes.iter().position(|a| a == id))
    {
        return Ok(Rides::generic(i));
    }
    if let Some(p) = tensor_prim(dtype) {
        return Ok(Rides::fixed(p));
    }
    Err(Error::new(
        dtype.span(),
        format!(
            "`{point}`: slot `{slot}` carries a dtype that is neither an axis of this method \
             nor fixed — write `Self::Tensor<T>` for `<T: Scalar>`, or `Self::Tensor<f32>`"
        ),
    ))
}

fn prim(ty: &Type) -> Option<Ident> {
    let Type::Path(p) = ty else { return None };
    let id = p.path.get_ident()?;
    let variant = match id.to_string().as_str() {
        "f32" => "F32",
        "i32" => "I32",
        "u32" => "U32",
        "bool" => "Bool",
        _ => return None,
    };
    Some(Ident::new(variant, id.span()))
}

fn tensor_prim(ty: &Type) -> Option<Ident> {
    if let Some(p) = prim(ty) {
        return Some(p);
    }
    let Type::Path(p) = ty else { return None };
    let id = p.path.get_ident()?;
    (id == "u8").then(|| Ident::new("U8", id.span()))
}

fn args(seg: &PathSegment) -> Vec<&Type> {
    match &seg.arguments {
        syn::PathArguments::AngleBracketed(a) => a
            .args
            .iter()
            .filter_map(|g| match g {
                syn::GenericArgument::Type(t) => Some(t),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}
