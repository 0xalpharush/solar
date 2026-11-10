"""Prove the rules of `evm-ir/peephole.isle` over physical EVM execution.

The reader follows Crocus (VanHattum et al., ASPLOS 2024): every ISLE term a rule
uses has a specification here, each rule is instantiated at every opcode, depth
and length its specifications admit, and each instance's left and right windows
are executed over symbolic words and compared. The specifications are:

* Window extractors (`lastN`, `unprotected_lastN`) bind the last N
  instructions. Boundary protection only restricts matching.
* Instruction facets: `push` binds a literal word, `any_push` and
  `push_block` push an arbitrary word, `opcode` binds a plain opcode or the
  one-byte form of a stack operation, and `dup`, `swap`, `exchange` and `pop`
  bind logical stack operations at every legal depth.
* Opcode classes are read from the opcode table in `crates/evmir/src/op.rs` and
  from the extractor bodies in `peephole/isle.rs`, which must match their
  transcriptions here, so a changed class fails closed.
* Variable-length extractors become families of windows, one per depth or
  operation they admit, with their outputs as functions of the window.
* Edits transcribe `Edit::apply` over the window's instructions.

Execution keeps the stack, byte-addressed memory (`MLOAD`, `MSTORE` in
`lean/EvmRules/Memory.lean`) and the ordered effects. Pure opcodes have the
word semantics of `Word.lean`. `JUMPI` records its destination and whether
it is taken; it must end its window, and a jump only one side keeps must be
untaken. Every other opcode is an opaque effect: both sides must perform the same
effects in the same order on equal operands and equal memory, and each result
and the memory after it are fresh inputs. Terminal opcodes and opcodes with
immediate-dependent stack effects are unsupported.

The obligation compares every stack word, the effects' operands and the final
memory; the replacement must read no deeper, grow no higher and leave the same
height. Instances that differ only in variable names share one theorem. Gas,
memory expansion, stack limits beyond these bounds, metadata and the Rust
extractors and edits remain trusted and are recorded by hash.
"""

import re
from functools import cache
from itertools import product

from .expr import MASK, SIGN, Cond, Expr, Unsupported
from .isle import ROOT, Rule, forms
from .lean import MEMORY, canonical
from .lean import OPERATIONS as WORD_OPERATIONS

OPCODE_TABLE = ROOT / "crates/evmir/src/op.rs"
EXTRACTORS = ROOT / "crates/codegen/src/backend/evm/ir/passes/peephole/isle.rs"
# Logical DUP and SWAP encode depths 1 to 235.
MAX_DEPTH = 235
# Rust bodies of the opcode classes the rules call, as transcribed below.
CLASS_BODIES = {
    "is_commutative": "op::is_commutative(opcode)",
    "flipped_comparison": "flipped_comparison(opcode)",
    "is_commutative_binop": "op::stack_io(opcode) == Some((2, 1)) && op::is_commutative(opcode)",
    "is_noncommutative_binop": "op::stack_io(opcode) == Some((2, 1)) && !op::is_commutative(opcode)",
    "is_sink": "op::stack_io(opcode) == Some((2, 0)) && !op::is_terminal(opcode) && opcode != JUMPI",
    "is_pure_binop": "op::is_pure(opcode) && op::stack_io(opcode) == Some((2, 1))",
    "is_deep_swap": "depth > 1",
    "u256_is_zero": "value.is_zero()",
    "u256_is_one": "value == U256::ONE",
}
SWAPPED_COMPARISONS = {"LT": "GT", "GT": "LT", "SLT": "SGT", "SGT": "SLT"}
SWAPPED_BODY = (
    "if is_commutative(opcode) { return Some(opcode); } Some(match opcode { "
    + " ".join(f"{a} => {b}," for a, b in SWAPPED_COMPARISONS.items())
    + " _ => return None, })"
)
FLIPPED_BODY = "op::swapped_binary_opcode(opcode).filter(|&flipped| flipped != opcode)"
# Guards that only price a rewrite; they never affect its equality obligation.
PRICE_GUARDS = {"complement_odd_dominates", "shift_pair_mask_profitable"}
# Pure opcodes the trusted evaluator folds, by operand count (`op/word.rs`).
FOLDED = {
    1: ("NOT", "CLZ", "ISZERO"),
    2: (
        "ADD", "SUB", "MUL", "DIV", "SDIV", "MOD", "SMOD", "EXP", "AND", "OR",
        "XOR", "SHL", "SHR", "SAR", "BYTE", "SIGNEXTEND", "LT", "GT", "SLT",
        "SGT", "EQ",
    ),
}  # fmt: skip

# Rules left unverified, by digest, with the reason. Reports list them as excluded, never as
# proved; a changed rule loses its entry, so it must prove or be excluded again.
EXCLUDED = {
    "a273043cfbd11e0a3261a1456839ccde0917a19411f467db6cd894add85991ea": (
        "Edit.ReloadStoredValue needs one more stack word than the window it replaces, "
        "so the rewrite can overflow a stack that the original fits"
    ),
    "9bb9fed10ed7f15ac4b23f859cbf024c7bac85533b634dc29985ee6649709d67": (
        "noop_stack_suffix matches unboundedly many push and stack-operation sequences, "
        "which need a general stack-machine theorem rather than instances"
    ),
}

# Trusted contracts of every peephole rule; reports list them with the proofs.
CONTRACTS = [
    "window extractors bind the block's last instructions; protection only restricts matching",
    "push binds literal words; any_push and push_block push arbitrary words",
    "opcode classes match the opcode table and their transcribed Rust bodies",
    "duplicates_known_zero: the word DUPn copies is zero",
    "fold_constants and fold_unary_constant evaluate their opcode as Word.lean does",
    "materialize_immediate pushes exactly its word within the guarded stack peak",
    "swap_pop_chain and swap_discard_chain match SWAPn POP*n SWAP1 POP and SWAPn POP*(n+1)",
    "invert_comparison's operands move but never copy, consume or inspect the pushed bound",
    "Edit::apply transcriptions; target lowering preserves stack operation semantics",
    "sufficient gas and stack; memory expansion, gas and metadata are not compared",
]


@cache
def opcode_table():
    """Each opcode's stack inputs and outputs and its traits, from the opcode table."""
    table = {}
    row = re.compile(
        r"^\s*0x[0-9a-f]{2} => (\w+) => \w+ => stack_io\(([^)]*)\) => traits\(([^)]*)\)",
        re.MULTILINE,
    )
    for name, io, traits in row.findall(OPCODE_TABLE.read_text()):
        counts = tuple(int(n) for n in io.split(",")) if "_" not in io else None
        table[name] = (counts, frozenset(t.strip() for t in traits.split("|") if t))
    if "ADD" not in table or "MSTORE" not in table:
        raise Unsupported("unreadable opcode table")
    return table


def rust_body(source, name):
    """The normalized body of the context method `name` in the extractor source."""
    found = re.search(
        rf"\n    fn {name}\(&mut self[^)]*\) -> [^{{]+\{{\n(.*?)\n    \}}\n",
        source,
        re.DOTALL,
    )
    if found is None:
        raise Unsupported(f"missing Rust extractor {name}")
    lines = (line.split("//", 1)[0] for line in found[1].splitlines())
    return " ".join(" ".join(lines).split())


@cache
def opcode_classes():
    """The opcodes of each class the rules call, read from the Rust sources."""
    table = opcode_table()
    source = EXTRACTORS.read_text()
    for name, body in CLASS_BODIES.items():
        if rust_body(source, name) != body:
            raise Unsupported(f"changed Rust opcode class: {name}")
    swapped = re.search(
        r"fn swapped_binary_opcode\(opcode: u8\) -> Option<u8> \{(.*?)\n\}\n",
        OPCODE_TABLE.read_text(),
        re.DOTALL,
    )
    if swapped is None or " ".join(swapped[1].split()) != SWAPPED_BODY:
        raise Unsupported("changed swapped_binary_opcode")
    free = re.search(
        r"\nfn flipped_comparison\(opcode: u8\) -> Option<u8> \{(.*?)\n\}\n",
        source,
        re.DOTALL,
    )
    if free is None or " ".join(free[1].split()) != FLIPPED_BODY:
        raise Unsupported("changed flipped_comparison")
    classes = {}
    for name in ("absorbs_zero", "zero_identity"):
        body = rust_body(source, name)
        match = re.fullmatch(r"matches!\(opcode, ([A-Z0-9_ |]+)\)", body)
        if match is None:
            raise Unsupported(f"changed Rust opcode class: {name}")
        classes[name] = {op.strip() for op in match[1].split("|")}

    def has(name, trait):
        return trait in table[name][1]

    def io(name):
        return table[name][0]

    classes["is_commutative"] = {n for n in table if has(n, "COMMUTATIVE")}
    classes["is_commutative_binop"] = {
        n for n in table if io(n) == (2, 1) and has(n, "COMMUTATIVE")
    }
    classes["is_noncommutative_binop"] = {
        n for n in table if io(n) == (2, 1) and not has(n, "COMMUTATIVE")
    }
    classes["is_sink"] = {
        n for n in table if io(n) == (2, 0) and not has(n, "TERMINAL") and n != "JUMPI"
    }
    classes["is_pure_binop"] = {n for n in table if has(n, "PURE") and io(n) == (2, 1)}
    unknown = set().union(*classes.values()) - set(table)
    if unknown:
        raise Unsupported(f"unknown opcodes in classes: {sorted(unknown)}")
    return classes


def flipped(name):
    """`flipped_comparison`: the operand-swapped opcode when it differs."""
    return SWAPPED_COMPARISONS.get(name)


def opcode_instruction(name):
    """The instruction an `opcode` facet or an edit's opcode names."""
    if name == "POP":
        return ("pop",)
    for kind in ("DUP", "SWAP"):
        if name.startswith(kind) and name[len(kind) :].isdecimal():
            depth = int(name[len(kind) :])
            if 1 <= depth <= 16:
                return (kind.lower(), depth)
    if name.startswith("PUSH"):
        raise Unsupported("the opcode facet never matches a push")
    return ("op", name)


class Machine:
    """Symbolic execution of one window over an arbitrary entry stack.

    Entry words are named by their depth below the entry top, `x0` first, and
    materialized lazily below everything the window pushed.
    """

    def __init__(self, table):
        self.table = table
        self.stack = []
        self.inputs = 0
        self.peak = 0
        self.memory = Expr.var(MEMORY)
        self.effects = []
        self.opaque = 0

    def need(self, depth):
        while len(self.stack) < depth:
            self.pad(self.inputs + 1)

    def pad(self, depth):
        while self.inputs < depth:
            self.stack.insert(0, Expr.var(f"x{self.inputs}"))
            self.inputs += 1

    @property
    def height(self):
        return len(self.stack) - self.inputs

    def pop(self):
        self.need(1)
        return self.stack.pop()

    def run(self, instructions):
        for instruction in instructions:
            if self.effects and self.effects[-1][0] == "JUMPI":
                raise Unsupported("JUMPI must end its window")
            self.step(instruction)
            self.peak = max(self.peak, self.height)
        return self

    def step(self, instruction):
        match instruction:
            case ("push", word) | ("materialize", word):
                self.stack.append(word)
            case ("dup", depth):
                self.need(depth)
                self.stack.append(self.stack[-depth])
            case ("swap", depth):
                self.need(depth + 1)
                stack = self.stack
                stack[-1], stack[-depth - 1] = stack[-depth - 1], stack[-1]
            case ("exchange", first, second):
                self.need(max(first, second) + 1)
                stack, a, b = self.stack, -first - 1, -second - 1
                stack[a], stack[b] = stack[b], stack[a]
            case ("pop",):
                self.pop()
            case ("body", depth):
                # The operands keep the protected word at `depth` with one independent
                # operand beside it; nothing else they compute depends on that word.
                protected = self.pop()
                operand = Expr.var("operand")
                self.stack += (
                    [operand, protected] if depth == 0 else [protected, operand]
                )
            case ("op", name):
                self.operation(name)
            case _:
                raise Unsupported(f"unmodeled instruction: {instruction}")

    def operation(self, name):
        io, traits = self.table.get(name, (None, frozenset()))
        if io is None or "TERMINAL" in traits:
            raise Unsupported(f"unmodeled opcode: {name}")
        inputs, outputs = io
        args = tuple(self.pop() for _ in range(inputs))
        lower = name.lower()
        if name == "MLOAD":
            self.stack.append(Expr("mload", (self.memory, args[0])))
        elif name == "MSTORE":
            self.memory = Expr("mstore", (self.memory, *args))
        elif name == "JUMPI":
            destination, condition = args
            taken = Expr("iszero", (Expr("iszero", (condition,)),))
            self.effects.append(("JUMPI", (destination, taken), None))
        elif "PURE" in traits and outputs == 1 and WORD_OPERATIONS.get(lower) == inputs:
            self.stack.append(Expr(lower, args))
        elif "PURE" in traits:
            raise Unsupported(f"pure opcode without word semantics: {name}")
        else:
            # Each result and the memory after the effect are fresh: both sides must
            # perform the same effects in the same order.
            index = self.opaque
            self.opaque += 1
            self.effects.append((name, args, self.memory))
            self.stack += [Expr.var(f"{name}:{index}:{j}") for j in range(outputs)]
            self.memory = Expr.var(f"{MEMORY}:{index}")


def compare(table, before, after):
    """The word difference and memory pair of two windows, with their stack shapes.

    The difference ORs the XOR of every pair of words that are not structurally
    identical, so it is zero exactly when both windows agree.
    """
    lhs, rhs = Machine(table).run(before), Machine(table).run(after)
    if rhs.inputs > lhs.inputs or rhs.peak > lhs.peak or rhs.height != lhs.height:
        raise Unsupported("replacement reads deeper, grows higher or changes height")
    rhs.pad(lhs.inputs)
    lhs.pad(lhs.inputs)
    words = list(zip(lhs.stack, rhs.stack))
    untaken = []
    kinds = [effect[0] for effect in lhs.effects]
    if kinds == [effect[0] for effect in rhs.effects]:
        pairs = list(zip(lhs.effects, rhs.effects))
    elif [k for k in kinds if k != "JUMPI"] == [e[0] for e in rhs.effects]:
        # A jump only the left window performs must never be taken.
        untaken = [e[1][1] for e in lhs.effects if e[0] == "JUMPI"]
        pairs = list(zip([e for e in lhs.effects if e[0] != "JUMPI"], rhs.effects))
    else:
        raise Unsupported("windows perform different effects")
    for (_, left, left_memory), (_, right, right_memory) in pairs:
        if left_memory != right_memory:
            raise Unsupported("an effect sees different memory")
        words += zip(left, right)
    difference = Expr.const(0)
    for a, b in words:
        if a != b:
            difference = Expr("or", (difference, Expr("xor", (a, b))))
    for taken in untaken:
        difference = Expr("or", (difference, taken))
    memory = None if lhs.memory == rhs.memory else (lhs.memory, rhs.memory)
    return {
        "minimum_stack": lhs.inputs,
        "peak_growth": lhs.peak,
        "difference": difference,
        "memory": memory,
    }


def peephole_rules(path):
    """Read the rules of a peephole file, skipping its declarations."""
    source = path.read_text()
    rules = [
        Rule(form, line, str(path)) for form, line in forms(source) if form[0] == "rule"
    ]
    if not rules:
        raise ValueError("peephole file contains no rules")
    return source, rules


def opcode_name(arg, bindings):
    if arg.startswith("$"):
        return arg[1:]
    name = bindings.get(arg)
    if not isinstance(name, str):
        raise Unsupported(f"unbound opcode: {arg}")
    return name


def depth_value(arg, bindings):
    value = int(arg) if arg.isdecimal() else bindings.get(arg)
    if not isinstance(value, int):
        raise Unsupported(f"unbound depth: {arg}")
    return value


def facet(pattern, bindings, index):
    """The instruction an instruction facet matches under `bindings`."""
    match pattern:
        case ("push", word):
            if word == "_":
                return ("push", Expr.var(f"pushed_{index}"))
            return ("push", bindings.get(word, Expr.var(f"pushed_{word}")))
        case ("any_push",):
            return ("push", Expr.var(f"pushed_{index}"))
        case ("push_block",):
            return ("push", Expr.var(f"label_{index}"))
        case ("opcode", name):
            return opcode_instruction(opcode_name(name, bindings))
        case ("dup", depth) | ("swap", depth):
            return (pattern[0], depth_value(depth, bindings))
        case ("exchange", first, second):
            return (
                "exchange",
                depth_value(first, bindings),
                depth_value(second, bindings),
            )
        case ("pop",):
            return ("pop",)
    raise Unsupported(f"unmodeled instruction facet: {pattern}")


def legal(instruction):
    match instruction:
        case ("dup", depth) | ("swap", depth):
            return 1 <= depth <= MAX_DEPTH
        case ("exchange", first, second):
            return 1 <= first < second and first + second <= 30
    return True


def family(name, outputs, table, classes):
    """Each window a variable-length extractor matches: (window, outputs, assumptions)."""
    words = [Expr.var(f"pushed_{i}") for i in range(3)]
    if name == "duplicates_known_zero" and not outputs:
        for depth in range(1, MAX_DEPTH + 1):
            # The duplicated word is the entry word `depth - 1` below the top.
            yield (
                [("dup", depth)],
                {},
                [Cond("eq", (Expr.var(f"x{depth - 1}"), Expr.const(0)))],
            )
    elif name in ("fold_constants", "fold_unary_constant") and len(outputs) == 1:
        arity = 2 if name == "fold_constants" else 1
        for opcode in FOLDED[arity]:
            pushed = words[:arity]
            # The evaluator takes its operands in pop order, the last push first.
            value = Expr(opcode.lower(), tuple(reversed(pushed)))
            window = [("push", word) for word in pushed] + [("op", opcode)]
            yield window, {outputs[0]: value}, []
    elif name == "negated_subtrahend" and len(outputs) == 1:
        constant = words[0]
        window = [("push", constant), ("swap", 1), ("op", "SUB")]
        yield window, {outputs[0]: Expr("sub", (Expr.const(0), constant))}, []
    elif name == "untaken_jump" and not outputs:
        yield (
            [("push", Expr.const(0)), ("push", Expr.var("label")), ("op", "JUMPI")],
            {},
            [],
        )
    elif name == "swap_pop_chain" and len(outputs) == 1:
        # The merged SWAP(n+1) must exist.
        for depth in range(1, MAX_DEPTH):
            window = [("swap", depth), *[("pop",)] * depth, ("swap", 1), ("pop",)]
            yield window, {outputs[0]: len(window)}, []
    elif name == "swap_discard_chain" and len(outputs) == 1:
        for depth in range(1, MAX_DEPTH + 1):
            window = [("swap", depth), *[("pop",)] * (depth + 1)]
            yield window, {outputs[0]: len(window)}, []
    elif name == "invert_comparison" and len(outputs) == 3:
        count, value, opposite = outputs
        constant = words[0]
        for opcode, depth in product(sorted(SWAPPED_COMPARISONS), (0, 1)):
            # Signed bounds are biased into unsigned order; the adjusted bound must not wrap.
            bias = Expr.const(SIGN if opcode in ("SLT", "SGT") else 0)
            ordered = Expr("xor", (constant, bias))
            if (opcode in ("GT", "SGT")) == (depth == 1):
                bound, limit = Expr("add", (ordered, Expr.const(1))), MASK
            else:
                bound, limit = Expr("sub", (ordered, Expr.const(1))), 0
            window = [
                ("push", constant),
                ("body", depth),
                ("op", opcode),
                ("op", "ISZERO"),
            ]
            yield (
                window,
                {
                    count: len(window),
                    value: Expr("xor", (bound, bias)),
                    opposite: flipped(opcode),
                },
                [Cond("ne", (ordered, Expr.const(limit)))],
            )
    else:
        raise Unsupported(f"unmodeled window extractor: {name}")


def edited(edit, window, bindings):
    """Transcribes `Edit::apply` on the matched window."""
    name, *args = edit
    name = name.removeprefix("Edit.")
    after = list(window)

    def opcode(arg):
        return opcode_instruction(opcode_name(arg, bindings))

    def word(arg):
        value = bindings.get(arg)
        if not isinstance(value, Expr):
            raise Unsupported(f"unbound edit word: {arg}")
        return value

    match name, args:
        case "Keep", [length] if length.isdecimal() and int(length) <= len(after):
            return after[: int(length)]
        case "OverwritePush0", []:
            after[0] = ("push", Expr.const(0))
        case "RemoveFirstKeepOne", []:
            return after[1:2]
        case "RemoveFirstKeepTwo", []:
            return after[1:3]
        case "RemoveFirstOverwrite", [code]:
            after = after[1:]
            after[0] = opcode(code)
        case "SwapOverwrite", [code]:
            after[0], after[1] = opcode(code), after[0]
        case "OverwriteOne", [code]:
            return [opcode(code)]
        case "OverwriteTwo", [code]:
            return [("swap", 1), opcode(code)]
        case "MergeSwapPop", []:
            after[0] = ("swap", len(after) - 2)
            after[-2] = ("pop",)
            return after[:-1]
        case "DropDiscardedSwap", []:
            return after[1:]
        case "FoldConstants", [value]:
            return [("materialize", word(value))]
        case "AddNegated", [value]:
            return [("materialize", word(value)), ("op", "ADD")]
        case "ZeroLessIszero", []:
            return after[2:]
        case "ReloadStoredValue", []:
            after[0], after[3] = after[3], after[0]
            after[1], after[2] = after[2], after[1]
            after[0] = opcode("$DUP1")
            return after[:3]
        case "DropNonzeroTest", []:
            after = after[2:]
            after[1] = opcode("$JUMPI")
        case "EqIszeroJumpi", []:
            after[0] = opcode("$SUB")
            del after[1]
            after[2] = opcode("$JUMPI")
        case "SwapPair", [first, second]:
            return [
                ("swap", depth_value(first, bindings)),
                ("swap", depth_value(second, bindings)),
            ]
        case "MoveConstantThroughSwaps", []:
            return [after[3], *after[:3]]
        case "InvertComparison", [value, code]:
            after[0] = ("push", word(value))
            return [*after[:-2], opcode(code)]
        case "ShiftPairMask", []:
            return [after[2], ("op", "SHL"), after[2], ("op", "SHR")]
        case _:
            raise Unsupported(f"unmodeled peephole edit: {edit}")
    return after


def last_class(root, instruction):
    """Whether the rule set `root` is tried for a window ending at `instruction`."""
    if root == "late_sub":
        return instruction == ("op", "SUB")
    if root == "late_mask":
        return instruction == ("op", "AND")
    kind = instruction[0]
    if kind in ("dup", "swap", "exchange", "pop"):
        return root == f"peep_{kind}"
    return kind == "op" and root == "peep_op"


def peephole_variants(rule, table=None, classes=None):
    """Return every instance of a rule with the obligations it must meet."""
    table = opcode_table() if table is None else table
    classes = opcode_classes() if classes is None else classes
    form = rule.form
    body = form[2:] if isinstance(form[1], str) and form[1].isdecimal() else form[1:]
    if len(body) < 2:
        raise Unsupported("unmodeled peephole rule shape")
    (root, window), *guards, rewrite = body
    if rewrite[0] != "rewrite" or len(rewrite) != 3:
        raise Unsupported("unmodeled peephole rewrite")
    _, extent, edit = rewrite
    if not root.startswith(("peep_", "late_")):
        raise Unsupported(f"unmodeled peephole root: {root}")
    name, *patterns = window
    fixed = re.fullmatch(r"(?:unprotected_)?last(\d)", name)
    literals, filters, derived = {}, [], []
    pushed = {p[1] for p in patterns if isinstance(p, tuple) and p[0] == "push"}
    for guard in guards:
        match guard:
            case ("if-let", "true", ("u256_is_zero" | "u256_is_one" as test, word)) if (
                word in pushed
            ):
                literals[word] = Expr.const(int(test == "u256_is_one"))
            case ("if-let", "true", (predicate, *_)) if predicate in PRICE_GUARDS:
                pass
            case ("if-let", "true", ("is_deep_swap", depth)):
                filters.append(lambda b, depth=depth: depth_value(depth, b) > 1)
            case ("if-let", "true", (predicate, code)) if predicate in classes:
                filters.append(
                    lambda b, p=predicate, c=code: opcode_name(c, b) in classes[p]
                )
            case ("if-let", result, ("flipped_comparison", code)) if (
                result.isidentifier()
            ):
                derived.append((result, code))
            case _:
                raise Unsupported(f"unmodeled peephole guard: {guard}")
    instances = []
    if fixed:
        if int(fixed[1]) != len(patterns) or not all(
            isinstance(p, tuple) for p in patterns
        ):
            raise Unsupported("unmodeled window extent")
        if not extent.isdecimal() or int(extent) != len(patterns):
            raise Unsupported("rewrite extent differs from the window")
        opcodes = sorted(
            {p[1] for p in patterns if p[0] == "opcode" and not p[1].startswith("$")}
        )
        depths = sorted(
            {
                arg
                for p in patterns
                if p[0] in ("dup", "swap", "exchange")
                for arg in p[1:]
                if not arg.isdecimal()
            }
        )
        if len(depths) > 2:
            raise Unsupported("peephole rule has more than two depth bindings")
        names = [n for n in table if not n.startswith("PUSH")]
        for values in product(names, repeat=len(opcodes)):
            for depth_values in product(range(1, MAX_DEPTH + 1), repeat=len(depths)):
                bindings = {
                    **literals,
                    **dict(zip(opcodes, values)),
                    **dict(zip(depths, depth_values)),
                }
                if not all(f(bindings) for f in filters):
                    continue
                window_instructions = [
                    facet(p, bindings, i) for i, p in enumerate(patterns)
                ]
                if all(legal(i) for i in window_instructions):
                    instances.append((bindings, window_instructions, []))
    else:
        if not all(isinstance(p, str) and p.isidentifier() for p in patterns):
            raise Unsupported("unmodeled window extractor outputs")
        for window_instructions, outputs, assumptions in family(
            name, patterns, table, classes
        ):
            if extent not in outputs and not (
                extent.isdecimal() and int(extent) == len(window_instructions)
            ):
                raise Unsupported("rewrite extent differs from the window")
            if extent in outputs and outputs[extent] != len(window_instructions):
                raise Unsupported("rewrite extent differs from the window")
            if all(f(outputs) for f in filters):
                instances.append((outputs, window_instructions, assumptions))
    variants = []
    variable_tail = (
        fixed and patterns[-1][0] == "opcode" and not patterns[-1][1].startswith("$")
    )
    for bindings, before, assumptions in instances:
        bindings = dict(bindings)
        for result, code in derived:
            bindings[result] = flipped(opcode_name(code, bindings))
        if None in bindings.values():
            continue
        if not last_class(root, before[-1]):
            # The matcher never tries this rule for an opcode of another class.
            if variable_tail:
                continue
            raise Unsupported(
                "the window's last instruction belongs to another rule set"
            )
        after = edited(edit, before, bindings)
        variant = compare(table, before, after)
        variant.update(
            bindings={
                k: v.text() if isinstance(v, Expr) else v for k, v in bindings.items()
            },
            assumptions=assumptions,
        )
        variants.append(variant)
    if not variants:
        raise Unsupported("no applicable peephole instances")
    return variants


def canonical_condition(cond, names):
    return Cond(
        cond.op,
        tuple(
            canonical(arg, names)
            if isinstance(arg, Expr)
            else canonical_condition(arg, names)
            if isinstance(arg, Cond)
            else arg
            for arg in cond.args
        ),
    )


def peephole_theorems(name, variants):
    """One theorem per distinct obligation shape: word differences, then memory."""
    shapes = {}
    for variant in variants:
        names = {}
        difference = canonical(variant["difference"], names)
        memory = variant["memory"]
        memory = memory and tuple(canonical(m, names) for m in memory)
        assumptions = tuple(
            canonical_condition(c, names) for c in variant["assumptions"]
        )
        if difference != Expr.const(0):
            shapes.setdefault((difference, Expr.const(0), assumptions), None)
        if memory is not None:
            shapes.setdefault((*memory, assumptions), None)
        if difference == Expr.const(0) and memory is None:
            shapes.setdefault((difference, Expr.const(0), ()), None)
    return [
        (f"{name}_{index}", lhs, rhs, list(assumptions))
        for index, (lhs, rhs, assumptions) in enumerate(shapes)
    ]
