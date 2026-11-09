"""A small IEC 61131-3 Structured Text subset, translated to Python.

Copied from the author's earlier weld-cell project and extended here with FOR loops.

Supported: one FUNCTION_BLOCK with VAR_INPUT / VAR_OUTPUT / VAR_IN_OUT / VAR / VAR CONSTANT,
types BOOL INT DINT REAL TIME TON R_TRIG and ARRAY[lo..hi] OF <elementary>, statements
assignment, IF/ELSIF/ELSE, FOR ... TO ... [BY ...] DO ... END_FOR, CASE (integer or constant labels, ELSE), FB calls with named
parameters, and expressions with OR XOR AND NOT, comparisons, + - * / MOD, unary minus,
ABS MIN MAX INT_TO_REAL REAL_TO_INT, TIME literals (T#1H2M3S4MS) and TRUE/FALSE.
Identifiers are case-insensitive, as in ST. TIME is held as integer milliseconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ELEMENTARY = {"BOOL", "INT", "DINT", "REAL", "TIME"}
FBTYPES = {"TON", "R_TRIG"}
KEYWORDS = {"IF", "THEN", "ELSIF", "ELSE", "END_IF", "CASE", "OF", "END_CASE", "AND", "OR", "XOR", "NOT", "MOD",
            "TRUE", "FALSE", "FUNCTION_BLOCK", "END_FUNCTION_BLOCK", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR",
            "CONSTANT", "END_VAR", "ARRAY", "FOR", "TO", "BY", "DO", "END_FOR"}
FUNCS = {"ABS": "abs", "MIN": "min", "MAX": "max", "INT_TO_REAL": "float", "REAL_TO_INT": "_r2i"}

TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)
  | (?P<comment>\(\*.*?\*\)|//[^\n]*)
  | (?P<time>T\#[0-9_.]+(?:H|MS|M|S)(?:[0-9_.]+(?:H|MS|M|S))*)
  | (?P<real>\d+\.\d+(?:[eE][+-]?\d+)?)
  | (?P<int>\d+)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>:=|<=|>=|<>|\.\.|[-+*/=<>()\[\],;:.])
""", re.VERBOSE | re.DOTALL | re.IGNORECASE)


class STError(Exception):
    pass


@dataclass
class Tok:
    kind: str
    text: str
    line: int

    @property
    def up(self) -> str:
        return self.text.upper()


def tokenize(src: str) -> list[Tok]:
    out, pos, line = [], 0, 1
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise STError(f"line {line}: unexpected {src[pos:pos + 12]!r}")
        kind = m.lastgroup
        text = m.group()
        if kind not in ("ws", "comment"):
            out.append(Tok(kind, text, line))
        line += text.count("\n")
        pos = m.end()
    return out


def time_ms(lit: str) -> int:
    total = 0.0
    for num, unit in re.findall(r"([0-9_.]+)(MS|H|M|S)", lit[2:].upper()):
        v = float(num.replace("_", ""))
        total += v * {"H": 3_600_000, "M": 60_000, "S": 1000, "MS": 1}[unit]
    return int(round(total))


@dataclass
class Var:
    name: str
    section: str          # input / output / inout / local / constant
    type: str             # elementary, FB type or ARRAY
    elem: str | None = None
    lo: int = 0
    hi: int = 0
    init: object = None


@dataclass
class Compiled:
    name: str
    vars: dict[str, Var]               # lower-case name -> Var
    python: str
    fn: object = field(repr=False, default=None)


# --- runtime FBs -----------------------------------------------------------------
class TON:
    def __init__(self):
        self.IN = False
        self.PT = 0
        self.Q = False
        self.ET = 0

    def __call__(self, IN, PT, dt):
        self.PT = int(PT)
        if IN:
            self.ET = min(self.ET + (dt if self.IN else 0), self.PT) if self.IN else 0
            self.Q = self.ET >= self.PT
        else:
            self.ET = 0
            self.Q = False
        self.IN = bool(IN)


class R_TRIG:  # noqa: N801  (IEC name)
    def __init__(self):
        self.CLK = False
        self.Q = False
        self._m = False

    def __call__(self, CLK):
        self.CLK = bool(CLK)
        self.Q = self.CLK and not self._m
        self._m = self.CLK


def _div(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return int(a / b)
    return a / b


def _mod(a, b):
    return a - b * int(a / b)


def _r2i(x):
    return int(round(x))


def _to_int(x):
    return int(x) if isinstance(x, bool) else int(x) if isinstance(x, int) else int(round(x))


# --- compiler ----------------------------------------------------------------------
class _Parser:
    def __init__(self, toks: list[Tok]):
        self.t = toks
        self.i = 0
        self.vars: dict[str, Var] = {}
        self.out: list[str] = []
        self.tmp = 0

    # token helpers
    def peek(self, k: int = 0) -> Tok | None:
        j = self.i + k
        return self.t[j] if j < len(self.t) else None

    def at(self, *words: str) -> bool:
        p = self.peek()
        return p is not None and p.up in words

    def take(self, *words: str) -> Tok:
        p = self.peek()
        if p is None:
            raise STError("unexpected end of source")
        if words and p.up not in words:
            raise STError(f"line {p.line}: expected {' or '.join(words)}, got {p.text!r}")
        self.i += 1
        return p

    # declarations
    def parse(self) -> Compiled:
        self.take("FUNCTION_BLOCK")
        name = self.take().text
        while self.at("VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR"):
            sec_tok = self.take().up
            section = {"VAR_INPUT": "input", "VAR_OUTPUT": "output", "VAR_IN_OUT": "inout", "VAR": "local"}[sec_tok]
            if sec_tok == "VAR" and self.at("CONSTANT"):
                self.take()
                section = "constant"
            while not self.at("END_VAR"):
                self.decl(section)
            self.take("END_VAR")
        self.emit_block(["END_FUNCTION_BLOCK"], 1)
        self.take("END_FUNCTION_BLOCK")
        body = "\n".join(self.out) or "    pass"
        return Compiled(name, self.vars, f"def _scan(s):\n{body}\n")

    def decl(self, section: str) -> None:
        names = [self.take().text]
        while self.at(","):
            self.take()
            names.append(self.take().text)
        self.take(":")
        tt = self.take()
        if tt.up == "ARRAY":
            self.take("[")
            lo = int(self.take().text)
            self.take("..")
            hi = int(self.take().text)
            self.take("]")
            self.take("OF")
            elem = self.take().up
            if elem not in ELEMENTARY:
                raise STError(f"line {tt.line}: arrays of {elem} not supported")
            typ = "ARRAY"
        else:
            typ, elem, lo, hi = tt.up, None, 0, 0
            if typ not in ELEMENTARY | FBTYPES:
                raise STError(f"line {tt.line}: type {tt.text} not supported")
        init = None
        if self.at(":="):
            self.take()
            init = self.const_value()
        self.take(";")
        for n in names:
            if n.lower() in self.vars:
                raise STError(f"line {tt.line}: {n} declared twice")
            if section == "constant" and init is None:
                raise STError(f"line {tt.line}: constant {n} needs a value")
            self.vars[n.lower()] = Var(n, section, typ, elem, lo, hi, init)

    def const_value(self):
        neg = False
        if self.at("-"):
            self.take()
            neg = True
        t = self.take()
        if t.kind == "time":
            v = time_ms(t.text)
        elif t.kind == "int":
            v = int(t.text)
        elif t.kind == "real":
            v = float(t.text)
        elif t.up in ("TRUE", "FALSE"):
            v = t.up == "TRUE"
        else:
            raise STError(f"line {t.line}: constant expected, got {t.text!r}")
        return -v if neg else v

    # statements
    def emit(self, line: str, ind: int) -> None:
        self.out.append("    " * ind + line)

    def emit_block(self, stop: list[str], ind: int, case_body: bool = False) -> None:
        start = len(self.out)
        while not self.at(*stop) and not (case_body and self.is_case_label()):
            self.statement(ind)
        if len(self.out) == start:
            self.emit("pass", ind)

    def is_case_label(self) -> bool:
        p, q = self.peek(), self.peek(1)
        if p is None or q is None:
            return False
        return (p.kind in ("int", "ident") and p.up not in KEYWORDS) and q.text in (":", ",")

    def statement(self, ind: int) -> None:
        t = self.peek()
        if t.text == ";":
            self.take()
            return
        if t.up == "IF":
            self.take()
            self.emit(f"if {self.expr()}:", ind)
            self.take("THEN")
            self.emit_block(["ELSIF", "ELSE", "END_IF"], ind + 1)
            while self.at("ELSIF"):
                self.take()
                self.emit(f"elif {self.expr()}:", ind)
                self.take("THEN")
                self.emit_block(["ELSIF", "ELSE", "END_IF"], ind + 1)
            if self.at("ELSE"):
                self.take()
                self.emit("else:", ind)
                self.emit_block(["END_IF"], ind + 1)
            self.take("END_IF")
            self.take(";")
            return
        if t.up == "FOR":
            self.take()
            ctr = self.var(self.take())
            if ctr.type not in ("INT", "DINT") or ctr.section == "constant":
                raise STError(f"line {t.line}: FOR counter {ctr.name} must be an INT/DINT variable")
            self.take(":=")
            a = self.expr()
            self.take("TO")
            b = self.expr()
            step = "1"
            if self.at("BY"):
                self.take()
                step = self.expr()
            self.take("DO")
            self.emit(f"for s.{ctr.name} in _for({a}, {b}, {step}):", ind)
            self.emit_block(["END_FOR"], ind + 1)
            self.take("END_FOR")
            self.take(";")
            return
        if t.up == "CASE":
            self.take()
            self.tmp += 1
            cv = f"_c{self.tmp}"
            self.emit(f"{cv} = {self.expr()}", ind)
            self.take("OF")
            first = True
            while self.is_case_label():
                labels = [self.label_value()]
                while self.at(","):
                    self.take()
                    labels.append(self.label_value())
                self.take(":")
                self.emit(f"{'if' if first else 'elif'} {cv} in ({', '.join(map(str, labels))},):", ind)
                first = False
                self.emit_block(["ELSE", "END_CASE"], ind + 1, case_body=True)
            if self.at("ELSE"):
                self.take()
                self.emit("else:" if not first else "if True:", ind)
                self.emit_block(["END_CASE"], ind + 1)
            self.take("END_CASE")
            self.take(";")
            return
        if t.kind == "ident" and t.up in KEYWORDS:
            raise STError(f"line {t.line}: unexpected keyword {t.text}")
        if t.kind == "ident":
            v = self.var(t)
            if v.type in FBTYPES and self.peek(1) and self.peek(1).text == "(":
                self.fb_call(v, ind)
                return
            target, ttype = self.lvalue()
            self.take(":=")
            e = self.expr()
            self.take(";")
            self.emit(f"{target} = {self.coerce(e, ttype)}", ind)
            return
        raise STError(f"line {t.line}: unexpected {t.text!r}")

    def label_value(self):
        t = self.take()
        if t.kind == "int":
            return int(t.text)
        v = self.vars.get(t.text.lower())
        if v is None or v.section != "constant":
            raise STError(f"line {t.line}: CASE label {t.text} must be an integer or constant")
        return v.init

    def fb_call(self, v: Var, ind: int) -> None:
        self.take()
        self.take("(")
        args = {}
        while not self.at(")"):
            pname = self.take().up
            self.take(":=")
            args[pname] = self.expr()
            if self.at(","):
                self.take()
        self.take(")")
        self.take(";")
        if v.type == "TON":
            if set(args) != {"IN", "PT"}:
                raise STError(f"{v.name}: TON needs IN and PT")
            self.emit(f"s.{v.name}({args['IN']}, {args['PT']}, s._dt)", ind)
        else:
            if set(args) != {"CLK"}:
                raise STError(f"{v.name}: R_TRIG needs CLK")
            self.emit(f"s.{v.name}({args['CLK']})", ind)

    def var(self, t: Tok) -> Var:
        v = self.vars.get(t.text.lower())
        if v is None:
            raise STError(f"line {t.line}: unknown identifier {t.text}")
        return v

    def lvalue(self) -> tuple[str, str]:
        t = self.take()
        v = self.var(t)
        if v.section == "constant":
            raise STError(f"line {t.line}: cannot assign constant {v.name}")
        if v.type == "ARRAY":
            self.take("[")
            idx = self.expr()
            self.take("]")
            return f"s.{v.name}[_ix(({idx}), {v.lo}, {v.hi})]", v.elem
        return f"s.{v.name}", v.type

    @staticmethod
    def coerce(e: str, typ: str) -> str:
        return {"BOOL": f"bool({e})", "INT": f"_to_int({e})", "DINT": f"_to_int({e})", "TIME": f"_to_int({e})",
                "REAL": f"float({e})"}[typ]

    # expressions (precedence: OR < XOR < AND < = <> < < > <= >= < + - < * / MOD < unary)
    def expr(self) -> str:
        e = self.xor()
        while self.at("OR"):
            self.take()
            e = f"({e} or {self.xor()})"
        return e

    def xor(self) -> str:
        e = self.and_()
        while self.at("XOR"):
            self.take()
            e = f"(bool({e}) != bool({self.and_()}))"
        return e

    def and_(self) -> str:
        e = self.eq()
        while self.at("AND"):
            self.take()
            e = f"({e} and {self.eq()})"
        return e

    def eq(self) -> str:
        e = self.rel()
        while self.at("=", "<>"):
            op = "==" if self.take().text == "=" else "!="
            e = f"({e} {op} {self.rel()})"
        return e

    def rel(self) -> str:
        e = self.add()
        while self.at("<", ">", "<=", ">="):
            op = self.take().text
            e = f"({e} {op} {self.add()})"
        return e

    def add(self) -> str:
        e = self.mul()
        while self.at("+", "-"):
            op = self.take().text
            e = f"({e} {op} {self.mul()})"
        return e

    def mul(self) -> str:
        e = self.unary()
        while self.at("*", "/", "MOD"):
            op = self.take().up
            r = self.unary()
            e = f"({e} * {r})" if op == "*" else f"_div({e}, {r})" if op == "/" else f"_mod({e}, {r})"
        return e

    def unary(self) -> str:
        if self.at("NOT"):
            self.take()
            return f"(not {self.unary()})"
        if self.at("-"):
            self.take()
            return f"(-{self.unary()})"
        return self.primary()

    def primary(self) -> str:
        t = self.take()
        if t.text == "(":
            e = self.expr()
            self.take(")")
            return e
        if t.kind == "int":
            return t.text
        if t.kind == "real":
            return t.text
        if t.kind == "time":
            return str(time_ms(t.text))
        if t.up in ("TRUE", "FALSE"):
            return "True" if t.up == "TRUE" else "False"
        if t.kind == "ident" and t.up in FUNCS and self.at("("):
            self.take("(")
            args = [self.expr()]
            while self.at(","):
                self.take()
                args.append(self.expr())
            self.take(")")
            return f"{FUNCS[t.up]}({', '.join(args)})"
        if t.kind == "ident":
            v = self.var(t)
            if v.section == "constant":
                return repr(v.init)
            if v.type == "ARRAY":
                self.take("[")
                idx = self.expr()
                self.take("]")
                return f"s.{v.name}[_ix(({idx}), {v.lo}, {v.hi})]"
            if v.type in FBTYPES:
                self.take(".")
                member = self.take().up
                allowed = {"TON": {"Q", "ET", "IN", "PT"}, "R_TRIG": {"Q", "CLK"}}[v.type]
                if member not in allowed:
                    raise STError(f"line {t.line}: {v.type} has no member {member}")
                return f"s.{v.name}.{member}"
            return f"s.{v.name}"
        raise STError(f"line {t.line}: unexpected {t.text!r} in expression")


def _for(a: int, b: int, step: int):
    if step == 0:
        raise STError("FOR with BY 0")
    return range(int(a), int(b) + (1 if step > 0 else -1), int(step))


def _ix(i: int, lo: int, hi: int) -> int:
    if not lo <= i <= hi:
        raise IndexError(f"array index {i} outside [{lo}..{hi}]")
    return i - lo


def compile_st(src: str) -> Compiled:
    c = _Parser(tokenize(src)).parse()
    ns = {"_div": _div, "_mod": _mod, "_r2i": _r2i, "_to_int": _to_int, "_ix": _ix, "_for": _for}
    exec(compile(c.python, f"<ST {c.name}>", "exec"), ns)  # noqa: S102  (our own generated code)
    c.fn = ns["_scan"]
    return c


def compile_file(path: Path) -> Compiled:
    return compile_st(Path(path).read_text(encoding="utf-8"))


class FBInstance:
    """One instance of a compiled function block. Access variables as attributes."""

    def __init__(self, c: Compiled):
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_dt", 0)
        for v in c.vars.values():
            if v.section == "constant":
                continue
            if v.type == "ARRAY":
                zero = {"BOOL": False, "REAL": 0.0}.get(v.elem, 0)
                val = [zero] * (v.hi - v.lo + 1)
            elif v.type == "TON":
                val = TON()
            elif v.type == "R_TRIG":
                val = R_TRIG()
            else:
                val = v.init if v.init is not None else {"BOOL": False, "REAL": 0.0}.get(v.type, 0)
            object.__setattr__(self, v.name, val)

    def __setattr__(self, key, value):
        v = self._c.vars.get(key.lower())
        if v is None:
            raise AttributeError(f"{self._c.name} has no variable {key}")
        if v.section == "constant":
            raise AttributeError(f"{key} is a constant")
        if v.type == "ARRAY":
            if len(value) > v.hi - v.lo + 1:
                raise ValueError(f"{key}: more than {v.hi - v.lo + 1} elements")
            arr = getattr(self, v.name)
            arr[:len(value)] = list(value)
            return
        object.__setattr__(self, v.name, value)

    def scan(self, dt_ms: int) -> None:
        object.__setattr__(self, "_dt", int(dt_ms))
        self._c.fn(self)
