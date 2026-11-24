use super::Channel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Q,
    K,
    V,
    O,
    GateUp,
    Down,
}

impl Site {
    pub fn bit(self) -> u32 {
        match self {
            Site::Q => 1 << 0,
            Site::K => 1 << 1,
            Site::V => 1 << 2,
            Site::O => 1 << 3,
            Site::GateUp => 1 << 4,
            Site::Down => 1 << 5,
        }
    }
}

pub struct Expr {
    pub(crate) kind: ExprKind,
}

pub(crate) enum ExprKind {
    X,
    Y,
    Mm(Channel, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Scale(Channel, Box<Expr>),
}

impl Expr {
    pub(crate) fn x() -> Self {
        Expr { kind: ExprKind::X }
    }
    pub(crate) fn y() -> Self {
        Expr { kind: ExprKind::Y }
    }
}

pub fn mm(w: &Channel, e: Expr) -> Expr {
    Expr {
        kind: ExprKind::Mm(*w, Box::new(e)),
    }
}

pub fn scale(e: Expr, l: &Channel) -> Expr {
    Expr {
        kind: ExprKind::Scale(*l, Box::new(e)),
    }
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr {
            kind: ExprKind::Add(Box::new(self), Box::new(rhs)),
        }
    }
}
