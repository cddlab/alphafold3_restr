# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Atom selection language parser for distance restraints.

Ported from Protenix with minor adaptations for AF3's atom layout.

Supported syntax:
  chain A B        - atoms in chain A or B
  resid 1 to 10   - atoms in residue (token) range 1-10
  resid 1 5 9     - atoms in specific residues (tokens)
  index 0 1 2     - atoms by global flat index
  and / or / not  - logical operators
  ( )             - grouping

Usage:
  selector = AtomSelector("chain A and resid 1 to 10")
  selector.matches({"chain": "A", "resid": 5, "index": 120})  # -> True

To match Protenix, `resid` corresponds to the 1-based token index.
"""
from __future__ import annotations

from typing import Dict, List, Union


class Chain:
  def __init__(self, names: List[str]):
    self.names = names

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    chain = mol.get('chain')
    return isinstance(chain, str) and chain in self.names


class ResId:
  def __init__(self, ids: List[int]):
    self.ids = ids

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    resid = mol.get('resid')
    return isinstance(resid, int) and resid in self.ids


class Index:
  def __init__(self, indices: List[int]):
    self.indices = indices

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    index = mol.get('index')
    return isinstance(index, int) and index in self.indices


class Not:
  def __init__(self, selection):
    self.selection = selection

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    return not self.selection.matches(mol)


class And:
  def __init__(self, selections: list):
    self.selections = selections

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    if not self.selections:
      return True
    return all(s.matches(mol) for s in self.selections)


class Or:
  def __init__(self, selections: list):
    self.selections = selections

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    if not self.selections:
      return False
    return any(s.matches(mol) for s in self.selections)


class Bracket:
  def __init__(self, selection):
    self.selection = selection

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    return self.selection.matches(mol)


class ParseError(ValueError):
  pass


RESERVED_KEYWORDS = {'and', 'or', 'not', 'to', 'resid', 'index', 'chain'}


class SelectionParser:
  def __init__(self, text: str):
    self.text = text
    self.pos = 0

  def _peek(self):
    return self.text[self.pos] if self.pos < len(self.text) else None

  def _consume_char(self, char: str):
    if self._peek() == char:
      self.pos += 1
      return char
    raise ParseError(f"Expected '{char}' at position {self.pos}, got '{self._peek()}'")

  def _consume_tag(self, tag: str):
    if self.text.startswith(tag, self.pos):
      if tag.isalpha() and (
          self.pos + len(tag) < len(self.text)
          and self.text[self.pos + len(tag)].isalnum()
      ):
        raise ParseError(
            f"Expected '{tag}' at position {self.pos}, got longer identifier"
        )
      self.pos += len(tag)
      return tag
    raise ParseError(f"Expected '{tag}' at position {self.pos}")

  def _skip_space0(self):
    while self.pos < len(self.text) and self.text[self.pos].isspace():
      self.pos += 1

  def _skip_space1(self):
    start_pos = self.pos
    self._skip_space0()
    if self.pos == start_pos:
      raise ParseError(f"Expected one or more spaces at position {self.pos}")

  def _parse_alphanumeric1(self) -> str:
    start_pos = self.pos
    if self.pos < len(self.text) and self.text[self.pos].isalnum():
      self.pos += 1
      while self.pos < len(self.text) and self.text[self.pos].isalnum():
        self.pos += 1
      return self.text[start_pos:self.pos]
    raise ParseError(f"Expected alphanumeric characters at position {self.pos}")

  def _parse_digit1(self) -> str:
    start_pos = self.pos
    if self.pos < len(self.text) and self.text[self.pos].isdigit():
      self.pos += 1
      while self.pos < len(self.text) and self.text[self.pos].isdigit():
        self.pos += 1
      return self.text[start_pos:self.pos]
    raise ParseError(f"Expected digits at position {self.pos}")

  def _parse_usize(self) -> int:
    try:
      return int(self._parse_digit1())
    except ValueError as e:
      raise ParseError(str(e))

  def _parse_identifier(self) -> str:
    identifier = self._parse_alphanumeric1()
    if identifier in {'and', 'or', 'not', 'to'}:
      raise ParseError(
          f"Identifier cannot be a reserved keyword: '{identifier}'"
          f" at position {self.pos - len(identifier)}"
      )
    return identifier

  def _parse_list_of_identifiers(self) -> List[str]:
    identifiers = [self._parse_identifier()]
    while True:
      saved_pos = self.pos
      try:
        self._skip_space1()
        identifiers.append(self._parse_identifier())
      except ParseError:
        self.pos = saved_pos
        break
    return identifiers

  def _parse_numbers(self) -> List[int]:
    first = self._parse_usize()
    saved_pos_for_to = self.pos
    try:
      self._skip_space1()
      self._consume_tag('to')
      self._skip_space1()
      last = self._parse_usize()
      if last < first:
        raise ParseError(f'Range end {last} is less than start {first}')
      return list(range(first, last + 1))
    except ParseError:
      self.pos = saved_pos_for_to
      numbers = [first]
      while True:
        saved_pos_loop = self.pos
        try:
          self._skip_space1()
          numbers.append(self._parse_usize())
        except ParseError:
          self.pos = saved_pos_loop
          break
      return numbers

  def _parse_resid(self):
    self._consume_tag('resid')
    self._skip_space1()
    return ResId(self._parse_numbers())

  def _parse_index(self):
    self._consume_tag('index')
    self._skip_space1()
    return Index(self._parse_numbers())

  def _parse_chain(self):
    self._consume_tag('chain')
    self._skip_space1()
    return Chain(self._parse_list_of_identifiers())

  def _parse_atom(self):
    for parser_func in [self._parse_chain, self._parse_resid, self._parse_index]:
      saved_pos = self.pos
      try:
        return parser_func()
      except ParseError:
        self.pos = saved_pos
    raise ParseError(f"Expected an atomic selection at position {self.pos}")

  def _parse_bracket(self):
    self._consume_char('(')
    expr = self.parse_expr()
    self._consume_char(')')
    return Bracket(expr)

  def _parse_primary(self):
    self._skip_space0()
    saved_pos = self.pos
    try:
      return self._parse_bracket()
    except ParseError:
      self.pos = saved_pos
      return self._parse_atom()

  def _parse_not(self):
    num_nots = 0
    while True:
      saved_pos = self.pos
      self._skip_space0()
      try:
        self._consume_tag('not')
        num_nots += 1
      except ParseError:
        self.pos = saved_pos
        break
    selection = self._parse_primary()
    for _ in range(num_nots):
      selection = Not(selection)
    return selection

  def _parse_and(self):
    operands = [self._parse_not()]
    while True:
      saved_pos = self.pos
      try:
        self._skip_space1()
        self._consume_tag('and')
        self._skip_space1()
        operands.append(self._parse_not())
      except ParseError:
        self.pos = saved_pos
        break
    return operands[0] if len(operands) == 1 else And(operands)

  def _parse_or(self):
    operands = [self._parse_and()]
    while True:
      saved_pos = self.pos
      try:
        self._skip_space1()
        self._consume_tag('or')
        self._skip_space1()
        operands.append(self._parse_and())
      except ParseError:
        self.pos = saved_pos
        break
    return operands[0] if len(operands) == 1 else Or(operands)

  def parse_expr(self):
    expr = self._parse_or()
    self._skip_space0()
    return expr

  def parse(self):
    parsed_node = self.parse_expr()
    if self.pos < len(self.text):
      raise ParseError(
          f"Unexpected trailing characters: '{self.text[self.pos:]}'"
          f" at position {self.pos}"
      )
    return parsed_node


class AtomSelector:
  """Parses and evaluates an atom selection string.

  The candidate atom dict should have:
    {"chain": str, "resid": int, "index": int}
  where:
    chain  = chain ID letter (e.g. "A", "B")
    resid  = residue number from the batch `residue_index` / `all_tokens.res_id`
    index  = global flat position index in the batch layout
  """

  def __init__(self, selection_string: str) -> None:
    self.selection_string = selection_string
    self.parsed_selection = SelectionParser(selection_string).parse()

  def matches(self, mol: Dict[str, Union[str, int]]) -> bool:
    return self.parsed_selection.matches(mol)
