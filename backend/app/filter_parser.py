"""Parse requested predicates without treating Boolean words as executable code."""

import re
from typing import Callable

from .models import FilterExpression, FilterPlan
from .schema_manager import SchemaManager


class FilterSyntaxError(ValueError):
    pass


class RequestedFilterParser:
    def __init__(self, schema: SchemaManager, resolve_field: Callable[[str], str | None]):
        self.schema = schema
        self.resolve_field = resolve_field

    def parse(self, clause: str) -> FilterPlan | FilterExpression:
        # Normalize comparison phrases before looking for Boolean connectors.
        phrases = (
            (r"(?:less|fewer)\s+than\s+or\s+equal\s+to|at\s+most", "<="),
            (r"(?:greater|more)\s+than\s+or\s+equal\s+to|at\s+least", ">="),
            (r"is\s+not\s+equal\s+to|is\s+not|does\s+not\s+equal|not\s+equal\s+to|isn't|isn’t", "!="),
            (r"greater\s+than|more\s+than|above|over", ">"),
            (r"less\s+than|fewer\s+than|below|under", "<"),
            (r"equal\s+to|equals|is", "="),
        )
        # Quoted category values retain their literal text.
        pieces = re.split(r"('(?:[^']*)'|\"(?:[^\"]*)\")", clause)
        for index in range(0, len(pieces), 2):
            pieces[index] = re.sub(r"\bis\s+(?=(?:less|greater|more|fewer|at|equal)\b)", "", pieces[index], flags=re.IGNORECASE)
            for pattern, operator in phrases:
                pieces[index] = re.sub(rf"\b(?:{pattern})\b", operator, pieces[index], flags=re.IGNORECASE)
        self.tokens = self._tokens("".join(pieces).strip().rstrip(".?!"))
        if self.tokens.count("(") > 64:
            raise FilterSyntaxError("The Boolean expression exceeds the supported grouping depth.")
        self.position = 0
        if not self.tokens:
            raise FilterSyntaxError("An explicit filter requires a predicate.")
        node = self._or()
        if self.position != len(self.tokens):
            raise FilterSyntaxError("The Boolean filter contains an unparsed constraint.")
        return node

    @staticmethod
    def _tokens(text: str) -> list[str]:
        tokens = []
        start = 0
        quote = None
        index = 0
        while index < len(text):
            character = text[index]
            if quote:
                if character == quote:
                    quote = None
                index += 1
                continue
            if character in {"'", '"'}:
                quote = character
                index += 1
                continue
            connector = re.match(r"\b(and|or)\b", text[index:], re.IGNORECASE)
            if character in "()" or (connector and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_"))):
                atom = text[start:index].strip()
                if atom:
                    tokens.append(atom)
                token = character if character in "()" else connector.group().lower()
                tokens.append(token)
                index += len(token)
                start = index
            else:
                index += 1
        if quote:
            raise FilterSyntaxError("The filter contains an unclosed quoted value.")
        if text[start:].strip():
            tokens.append(text[start:].strip())
        return tokens

    def _or(self):
        nodes = [self._and()]
        while self._consume("or"):
            nodes.append(self._and())
        return combine_filters("or", nodes)

    def _and(self):
        nodes = [self._primary()]
        while self._consume("and"):
            nodes.append(self._primary())
        return combine_filters("and", nodes)

    def _consume(self, token: str) -> bool:
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False

    def _primary(self):
        if self._consume("("):
            node = self._or()
            if not self._consume(")"):
                raise FilterSyntaxError("The Boolean filter contains unbalanced parentheses.")
            return node
        if self.position >= len(self.tokens) or self.tokens[self.position] in {"and", "or", ")"}:
            raise FilterSyntaxError("Every Boolean connector requires two operands.")
        atom = self.tokens[self.position]
        self.position += 1
        return self._atom(atom)

    def _atom(self, atom: str) -> FilterPlan:
        atom = re.sub(r"^(?:in|for|from|within)\s+", "", atom, flags=re.IGNORECASE).strip()
        match = re.fullmatch(
            r"([a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)*?)\s*"
            r"(>=|<=|!=|==|>|<|=|\bcontains\b)\s*(.+)", atom, re.IGNORECASE,
        )
        if not match:
            resolved = self.schema.resolve_categorical_phrase(atom, self.resolve_field)
            if resolved:
                return FilterPlan(column=resolved[0], operator="=", value=resolved[1])
            raise FilterSyntaxError(f"Unparsed filter constraint: {atom}")
        raw_field, operator, raw_value = match.groups()
        column = self.resolve_field(raw_field) or self.schema.resolve_business_term(raw_field)
        if not column:
            raise FilterSyntaxError(f"Unknown filter field: {raw_field}")
        if column not in self.schema.available_columns()["sales"] and column != "revenue":
            raise FilterSyntaxError(f"The requested filter field is unavailable in the sales data: {raw_field}")
        operator = "=" if operator == "==" else operator.lower()
        value = raw_value.strip().strip("\"'")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
            if not self.schema.is_numeric_field(column):
                raise FilterSyntaxError(f"A numeric comparison requires a numeric field: {raw_field}")
            number = float(value)
            value = int(number) if number.is_integer() else number
        elif operator == "contains":
            if column not in self.schema.categorical_index():
                raise FilterSyntaxError(f"Contains requires a categorical field: {raw_field}")
        elif operator in {"=", "!="}:
            candidates = self.schema.categorical_index().get(column, [])
            matches = [candidate for candidate in candidates if candidate.casefold() == value.casefold()]
            if len(matches) != 1:
                raise FilterSyntaxError(f"Unresolved categorical predicate: {raw_field} {operator} {value}")
            value = matches[0]
        else:
            raise FilterSyntaxError(f"Unresolved filter predicate: {atom}")
        return FilterPlan(column=column, operator=operator, value=value)


def combine_filters(op: str, nodes: list[FilterPlan | FilterExpression]):
    if len(nodes) == 1:
        return nodes[0]
    predicates = []
    groups = []
    for node in nodes:
        if isinstance(node, FilterPlan):
            predicates.append(node)
        elif node.op == op:
            predicates.extend(node.predicates)
            groups.extend(node.groups)
        else:
            groups.append(node)
    return FilterExpression(op=op, predicates=predicates, groups=groups)


def filter_signature(node: FilterPlan | FilterExpression | None):
    """Canonicalize associativity/commutativity without changing AND/OR meaning."""
    if node is None:
        return None
    if isinstance(node, FilterPlan):
        value = node.value
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return ("predicate", node.column, node.operator, repr(value))
    children = [*node.predicates, *node.groups]
    flattened = []
    for child in children:
        signature = filter_signature(child)
        if isinstance(child, FilterExpression) and child.op == node.op:
            flattened.extend(signature[1])
        else:
            flattened.append(signature)
    return (node.op, tuple(sorted(set(flattened), key=repr)))
