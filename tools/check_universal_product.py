#!/usr/bin/env python3

import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    category: str
    detail: str

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return f"{shown}:{self.line}: {self.category}: {self.detail}"


_MODEL_CATEGORY = re.compile(
    r"(?:Qwen|Gemma|Llama|Phi|Mistral|Mixtral|Model(?:Family|Type|Kind|Class|Category|Selector)|"
    r"(?:Architecture|Arch)(?:Type|Kind|Class|Category|Selector)|NamedArchitecture|"
    r"(?:Semantic|Dense|Moe|MoE|Recurrent|Attention|Routed)Layer|(?:Dense|Moe|MoE|Recurrent)Graph|"
    r"model_(?:family|type|kind)|architecture|arch_(?:type|kind)|layer_(?:type|kind|class))"
)
_SOURCE_FORMAT = re.compile(
    r"(?:GGMLType|GGUFType|SourceTypeId|source_type_id|(?:Structural|MetalWeightFormat)[A-Za-z0-9_]*(?:Q[1-8]|IQ[1-4])|"
    r"(?:^|_)(?:Q[1-8](?:_[A-Za-z0-9]+|K)|IQ[1-4](?:_[A-Za-z0-9]+|[A-Z]{2,}))(?=$|_))"
)
_RENAMED_CATEGORY = re.compile(
    r"(?:LoweringStrategy|ExecutionShape|LayerShape|GraphShape|PipelineRecipe|ProgramBundle|"
    r"BundleGroupKind|Capability(?:Type|Kind|Class|Selector|Flag))"
)
_SOURCE_TENSOR = re.compile(
    r"(?:source_)?tensor_(?:name|spelling|role|selector)|source_tensor", re.IGNORECASE
)
_PROVENANCE = re.compile(
    r"(?:^|_)(?:artifact(?:_?(?:id|path|hash|digest))?|fixture(?:_?id)?|"
    r"benchmark(?:_?id)?|golden(?:_?(?:id|hash|digest))?|provenance|source_?path|"
    r"file_?path|path_?name|path)(?:$|_)"
)
_ROUTE_SELECTION = re.compile(
    r"(?:route|select|dispatch|strategy|recipe|shape|capabil|bundle|implementation|"
    r"kernel|policy|executor|schedule|architecture|model_type|model_kind)", re.IGNORECASE
)
_BOUNDARY_VARIABLES = {
    "LAPLACE_PRODUCT_SOURCES",
    "LAPLACE_QUALIFICATION_ONLY_SOURCES",
    "LAPLACE_PRODUCT_TARGETS",
}
_SYSTEM_INCLUDE_ALLOWLIST = {
    "Accelerate/Accelerate.h",
    "CommonCrypto/CommonDigest.h",
    "CoreFoundation/CoreFoundation.h",
    "IOKit/ps/IOPSKeys.h",
    "IOKit/ps/IOPowerSources.h",
    "Metal/Metal.h",
    "MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h",
    "algorithm",
    "arm_neon.h",
    "array",
    "atomic",
    "bit",
    "cctype",
    "cerrno",
    "cfenv",
    "charconv",
    "chrono",
    "climits",
    "cmath",
    "cstddef",
    "cstdint",
    "cstdio",
    "cstdlib",
    "cstring",
    "deque",
    "dirent.h",
    "dispatch/dispatch.h",
    "exception",
    "fcntl.h",
    "fstream",
    "functional",
    "iostream",
    "iterator",
    "limits",
    "map",
    "memory",
    "metal_simdgroup_matrix",
    "metal_stdlib",
    "metal_tensor",
    "mutex",
    "new",
    "notify.h",
    "numeric",
    "optional",
    "os/os_sync_wait_on_address.h",
    "pthread/qos.h",
    "queue",
    "random",
    "set",
    "span",
    "sstream",
    "stdexcept",
    "string",
    "string_view",
    "sys/clonefile.h",
    "sys/mman.h",
    "sys/mount.h",
    "sys/stat.h",
    "sys/sysctl.h",
    "thread",
    "tuple",
    "type_traits",
    "unistd.h",
    "unordered_map",
    "unordered_set",
    "utility",
    "variant",
    "vector",
}
_EXTERNAL_STRING = re.compile(
    r"(?:Qwen|Gemma|Llama|Phi|Mistral|Mixtral|Q[1-8]_[A-Za-z0-9]+|IQ[1-4]_[A-Za-z0-9]+|"
    r"GGML|GGUF|fixture|benchmark|(?:blk|layers?)\.[0-9]+[^\"]*(?:weight|bias))",
    re.IGNORECASE
)
_RESERVED_DIAGNOSTIC_SINKS = {"fprintf", "stderr"}
_MACRO_DIRECTIVES = {
    "define", "undef", "if", "ifdef", "ifndef", "elif", "elifdef", "elifndef",
}


def _lex_cpp(source: str) -> list[Token]:
    tokens: list[Token] = []
    index = 0
    line = 1
    length = len(source)
    while index < length:
        char = source[index]
        if source.startswith("\\\r\n", index):
            line += 1
            index += 3
            continue
        if source.startswith("\\\n", index):
            line += 1
            index += 2
            continue
        if char.isspace():
            line += char == "\n"
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            if newline < 0:
                break
            index = newline
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                end = length - 2
            line += source.count("\n", index, end + 2)
            index = end + 2
            continue

        start_line = line
        raw_match = re.match(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(', source[index:])
        if raw_match:
            delimiter = raw_match.group(1)
            body_start = index + raw_match.end()
            marker = ")" + delimiter + '"'
            end = source.find(marker, body_start)
            end = length if end < 0 else end + len(marker)
            text = source[index:end]
            tokens.append(Token("string", text, start_line))
            line += text.count("\n")
            index = end
            continue

        prefix = re.match(r'(?:u8|u|U|L|@)?(?=["\'])', source[index:])
        quote_index = index + (prefix.end() if prefix else 0)
        if quote_index < length and source[quote_index] in "\"'":
            quote = source[quote_index]
            cursor = quote_index + 1
            escaped = False
            while cursor < length:
                current = source[cursor]
                if current == "\n":
                    line += 1
                if not escaped and current == quote:
                    cursor += 1
                    break
                if not escaped and current == "\\":
                    escaped = True
                else:
                    escaped = False
                cursor += 1
            tokens.append(Token("string" if quote == '"' else "char",
                                source[index:cursor], start_line))
            index = cursor
            continue

        identifier = re.match(r"[A-Za-z_][A-Za-z0-9_]*", source[index:])
        if identifier:
            text = identifier.group(0)
            tokens.append(Token("identifier", text, line))
            index += len(text)
            continue
        number = re.match(r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)", source[index:])
        if number:
            text = number.group(0)
            tokens.append(Token("number", text, line))
            index += len(text)
            continue
        two = source[index:index + 2]
        if two in {"==", "!=", "<=", ">=", "->", "::", "&&", "||"}:
            tokens.append(Token("symbol", two, line))
            index += 2
        else:
            tokens.append(Token("symbol", char, line))
            index += 1
    return tokens


def _statement(tokens: list[Token], index: int) -> list[Token]:
    first = index
    while first > 0 and tokens[first - 1].text not in {";", "{", "}"}:
        first -= 1
    last = index + 1
    while last < len(tokens) and tokens[last].text not in {";", "{", "}"}:
        last += 1
    return tokens[first:last]


def _string_payload(text: str) -> str:
    raw = re.match(
        r'^(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\((.*)\)\1"$',
        text,
        re.DOTALL,
    )
    if raw:
        return raw.group(2)
    quote = text.find('"')
    if quote < 0 or not text.endswith('"'):
        return text
    body = text[quote + 1:-1]
    simple = {
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        "'": "'",
        '"': '"',
        "?": "?",
    }

    def decode_escape(match: re.Match[str]) -> str:
        escape = match.group(1)
        if escape in {"\n", "\r\n"}:
            return ""
        if escape in simple:
            return simple[escape]
        if escape.startswith("x"):
            value = int(escape[1:], 16)
        elif escape.startswith("u") or escape.startswith("U"):
            value = int(escape[1:], 16)
        elif escape[0] in "01234567":
            value = int(escape, 8)
        else:
            return escape
        try:
            return chr(value)
        except ValueError:
            return escape

    return re.sub(
        r"\\(\r?\n|x[0-9A-Fa-f]+|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[0-7]{1,3}|.)",
        decode_escape,
        body,
        flags=re.DOTALL,
    )


def _reserved_sink_macro_tokens(tokens: list[Token], source: str) -> list[Token]:
    lines = source.splitlines()
    reserved: list[Token] = []
    for index, token in enumerate(tokens[:-1]):
        directive = tokens[index + 1]
        if (token.text != "#" or directive.kind != "identifier" or
                directive.line != token.line or directive.text not in _MACRO_DIRECTIVES):
            continue
        end_line = token.line
        while end_line <= len(lines) and lines[end_line - 1].rstrip().endswith("\\"):
            end_line += 1
        cursor = index + 2
        while cursor < len(tokens) and tokens[cursor].line <= end_line:
            candidate = tokens[cursor]
            if (candidate.kind == "identifier" and
                    candidate.text in _RESERVED_DIAGNOSTIC_SINKS):
                reserved.append(candidate)
            cursor += 1
    return reserved


def _scan_source(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tokens = _lex_cpp(source)
    found: dict[str, Finding] = {}

    def add(category: str, token: Token, detail: str) -> None:
        found.setdefault(category, Finding(path, token.line, category, detail))

    def provenance_selector(statement: list[Token], suspicious: Token) -> bool:
        names = [item.text for item in statement if item.kind == "identifier"]
        hard_identity = any(re.search(r"fixture|benchmark|golden|provenance", name,
                                      re.IGNORECASE) for name in names)
        routed = any(_ROUTE_SELECTION.search(name) for name in names)
        literal_identity = (re.search(r"path|hash|digest", suspicious.text,
                                      re.IGNORECASE) and
                            any(item.kind in {"number", "string"} for item in statement))
        return hard_identity or routed or bool(literal_identity)

    reserved_sink_macros = _reserved_sink_macro_tokens(tokens, source)
    for macro_token in reserved_sink_macros:
        add("reserved-diagnostic-sink-macro", macro_token,
            f"preprocessor directive touches reserved sink `{macro_token.text}`")

    for index, token in enumerate(tokens):
        if token.kind == "identifier":
            if _MODEL_CATEGORY.search(token.text):
                add("model-category", token,
                    f"category-bearing identifier `{token.text}`")
            if _SOURCE_FORMAT.search(token.text):
                add("source-format-selector", token,
                    f"source-format identifier `{token.text}`")
            if _SOURCE_TENSOR.search(token.text):
                add("source-tensor-selector", token,
                    f"source tensor identifier `{token.text}`")
            if _RENAMED_CATEGORY.search(token.text):
                add("renamed-category", token,
                    f"strategy/shape/capability/bundle category `{token.text}`")
            if re.search(r"(?:match|validate|derive).*(?:dense|moe|recurrent).*(?:graph|layer|edges|shape)",
                         token.text, re.IGNORECASE):
                add("fixed-graph-matcher", token,
                    f"serialized graph matcher `{token.text}`")
        elif token.kind == "string":
            if index > 0 and tokens[index - 1].kind == "string":
                continue
            end = index + 1
            while end < len(tokens) and tokens[end].kind == "string":
                end += 1
            translated = "".join(_string_payload(item.text)
                                 for item in tokens[index:end])
            if _EXTERNAL_STRING.search(translated):
                same_line = [item for item in tokens if item.line == token.line]
                is_include = any(item.text == "include" for item in same_line)
                if not is_include:
                    add("source-format-selector", token,
                        "external name appears in a product string")

        if token.text in {"==", "!="}:
            statement = _statement(tokens, index)
            names = [item for item in statement
                     if item.kind == "identifier"]
            suspicious = next((item for item in names
                               if item.text[:1].islower() and
                               _PROVENANCE.search(item.text)), None)
            if suspicious and provenance_selector(statement, suspicious):
                add("provenance-selector", suspicious,
                    f"provenance value `{suspicious.text}` controls behavior")

        if token.kind == "identifier" and token.text in {"operators", "operator_sequence"}:
            following = tokens[index + 1:index + 8]
            if (len(following) >= 2 and following[0].text == "[" and
                    following[1].kind == "number"):
                add("fixed-graph-matcher", token,
                    "serialized operator position controls behavior")
            joined = " ".join(item.text for item in following)
            if re.search(r"\. size \( \) (?:==|!=|<=|>=|<|>) [0-9]+", joined):
                add("fixed-graph-matcher", token,
                    "fixed operator cardinality controls behavior")

        if token.text == "switch":
            window = tokens[index:min(len(tokens), index + 20)]
            selector = " ".join(item.text for item in window)
            suspicious = next((item for item in window
                               if item.kind == "identifier" and
                               item.text[:1].islower() and
                               _PROVENANCE.search(item.text)), None)
            if suspicious:
                add("provenance-selector", suspicious,
                    f"provenance value `{suspicious.text}` controls behavior")
            if re.search(r"strategy|recipe|shape|capabil|bundle", selector,
                         re.IGNORECASE):
                add("renamed-category", token,
                    "switch selects a strategy/recipe/shape/capability/bundle category")

    return list(found.values())


def _local_includes(path: Path, root: Path) -> tuple[list[Path], list[Finding]]:
    macros: dict[str, tuple[str, bool]] = {}
    includes: list[Path] = []
    findings: list[Finding] = []
    in_block_comment = False
    for line_number, original in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = original
        if in_block_comment:
            end = line.find("*/")
            if end < 0:
                continue
            line = line[end + 2:]
            in_block_comment = False
        while "/*" in line:
            start = line.find("/*")
            end = line.find("*/", start + 2)
            if end < 0:
                line = line[:start]
                in_block_comment = True
                break
            line = line[:start] + line[end + 2:]
        line = re.sub(r"//.*$", "", line)
        define = re.match(
            r"\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+([\"<])([^\">]+)[\">]",
            line,
        )
        if define:
            macros[define.group(1)] = (define.group(3), define.group(2) == "<")
            continue
        include = re.match(r"\s*#\s*include\s+(.+?)\s*$", line)
        if not include:
            continue
        operand = include.group(1).strip()
        system = False
        if operand.startswith('"') and operand.endswith('"'):
            name = operand[1:-1]
        elif operand.startswith("<") and operand.endswith(">"):
            name = operand[1:-1]
            system = True
        elif operand in macros:
            name, system = macros[operand]
        else:
            findings.append(Finding(path, line_number, "include-error",
                                    f"unresolved include macro `{operand}`"))
            continue
        candidates = (path.parent / name, root / name, root / "src" / name)
        resolved = next((candidate.resolve() for candidate in candidates
                         if candidate.is_file()), None)
        if resolved is None:
            if system and name in _SYSTEM_INCLUDE_ALLOWLIST:
                continue
            if system:
                findings.append(Finding(
                    path, line_number, "include-error",
                    f"unresolved non-allowlisted angle include `{name}`",
                ))
            else:
                findings.append(Finding(path, line_number, "include-error",
                                        f"missing repository include `{name}`"))
            continue
        if resolved != root and root not in resolved.parents:
            findings.append(Finding(path, line_number, "include-error",
                                    f"include escapes repository root: `{name}`"))
            continue
        includes.append(resolved)
    return includes, findings


def _cmake_commands(text: str) -> list[tuple[str, list[str], int]]:
    commands: list[tuple[str, list[str], int]] = []
    index = 0
    line = 1
    while index < len(text):
        if text[index] == "\n":
            line += 1
            index += 1
            continue
        if text[index].isspace():
            index += 1
            continue
        if text[index] == "#":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            continue
        name_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
        if not name_match:
            index += 1
            continue
        name = name_match.group(0)
        command_line = line
        cursor = index + len(name)
        while cursor < len(text) and text[cursor].isspace():
            line += text[cursor] == "\n"
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            index = cursor
            continue
        body_start = cursor + 1
        cursor = body_start
        depth = 1
        quoted = False
        escaped = False
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "\n":
                line += 1
            if not quoted and char == "#":
                newline = text.find("\n", cursor)
                cursor = len(text) if newline < 0 else newline
                continue
            if char == '"' and not escaped:
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
                if depth == 0:
                    break
            escaped = char == "\\" and not escaped
            cursor += 1
        if depth != 0:
            raise ValueError(f"unterminated CMake command `{name}` at line {command_line}")
        body = text[body_start:cursor]
        commands.append((name.lower(), shlex.split(body, comments=False, posix=True),
                         command_line))
        index = cursor + 1
    return commands


def _cmake_list(catalog: Path, variable: str) -> list[str]:
    commands = _cmake_commands(catalog.read_text(encoding="utf-8"))
    unparsed = [command for command in commands
                if command[0] != "set" or not command[1] or
                command[1][0] not in _BOUNDARY_VARIABLES]
    if unparsed:
        command = unparsed[0]
        raise ValueError(f"unparsed catalog command {command[0]}() at line {command[2]}")
    declarations = [command for command in commands
                    if command[0] == "set" and command[1] and
                    command[1][0] == variable]
    mutations = [command for command in commands
                 if variable in command[1] and command not in declarations]
    if len(declarations) != 1:
        raise ValueError(f"{variable} must have exactly one literal set() declaration")
    if mutations:
        command = mutations[0]
        raise ValueError(f"unparsed mutation of {variable} by {command[0]}() at line {command[2]}")
    values = declarations[0][1][1:]
    if any("${" in value or "$<" in value for value in values):
        raise ValueError(f"{variable} must contain explicit source paths")
    return values


def scan_catalog(catalog: Path) -> list[Finding]:
    root = catalog.parent.parent.resolve()
    try:
        product = _cmake_list(catalog, "LAPLACE_PRODUCT_SOURCES")
        qualification = _cmake_list(catalog, "LAPLACE_QUALIFICATION_ONLY_SOURCES")
        targets = _cmake_list(catalog, "LAPLACE_PRODUCT_TARGETS")
    except (OSError, ValueError) as error:
        return [Finding(catalog, 1, "catalog-error", str(error))]

    if not targets or len(targets) != len(set(targets)) or "laplace" not in targets:
        return [Finding(catalog, 1, "catalog-error",
                        "LAPLACE_PRODUCT_TARGETS must contain unique targets including laplace")]
    overlap = sorted(set(product) & set(qualification))
    if overlap:
        return [Finding(catalog, 1, "catalog-error",
                        "product and qualification-only lists overlap: " + ", ".join(overlap))]

    findings: list[Finding] = []
    product_paths: list[Path] = []
    seen_product: set[Path] = set()
    for relative in product:
        path = (root / relative).resolve()
        if path in seen_product:
            findings.append(Finding(catalog, 1, "catalog-error",
                                    f"duplicate product source: {relative}"))
            continue
        seen_product.add(path)
        if root not in path.parents or path.suffix not in {".c", ".cc", ".cpp", ".m", ".mm"}:
            findings.append(Finding(catalog, 1, "catalog-error",
                                    f"invalid product source path: {relative}"))
        elif not path.is_file():
            findings.append(Finding(catalog, 1, "catalog-error",
                                    f"missing product source: {relative}"))
        else:
            product_paths.append(path)

    pending = list(product_paths)
    scanned: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in scanned:
            continue
        scanned.add(path)
        findings.extend(_scan_source(path))
        included, include_findings = _local_includes(path, root)
        findings.extend(include_findings)
        pending.extend(included)
    return sorted(findings, key=lambda finding: (str(finding.path), finding.line,
                                                  finding.category))


def verify_target_closure(catalog: Path, snapshot: Path) -> list[Finding]:
    root = catalog.parent.parent.resolve()
    try:
        declared_sources = set(_cmake_list(catalog, "LAPLACE_PRODUCT_SOURCES"))
        qualification = set(_cmake_list(catalog, "LAPLACE_QUALIFICATION_ONLY_SOURCES"))
        declared_targets = _cmake_list(catalog, "LAPLACE_PRODUCT_TARGETS")
        lines = snapshot.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError) as error:
        return [Finding(snapshot, 1, "closure-error", str(error))]

    findings: list[Finding] = []
    actual_sources: set[str] = set()
    actual_targets: list[str] = []
    links: dict[str, list[str]] = {}
    link_metadata: dict[tuple[str, str], tuple[str, str]] = {}
    current_target = ""
    for line_number, line in enumerate(lines, 1):
        if "\t" not in line:
            findings.append(Finding(snapshot, line_number, "closure-error",
                                    "malformed target-closure record"))
            continue
        kind, value = line.split("\t", 1)
        if kind == "target":
            current_target = value
            actual_targets.append(value)
            links.setdefault(value, [])
        elif kind == "source":
            if not current_target or not value or "$<" in value:
                findings.append(Finding(snapshot, line_number, "closure-error",
                                        f"unparsed target source `{value}`"))
                continue
            path = Path(value)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if root not in resolved.parents:
                findings.append(Finding(snapshot, line_number, "closure-error",
                                        f"target source escapes repository: `{value}`"))
                continue
            actual_sources.add(str(resolved.relative_to(root)))
        elif kind == "link":
            if not current_target or not value or "$<" in value:
                findings.append(Finding(snapshot, line_number, "closure-error",
                                        f"unparsed target link `{value}`"))
            else:
                links[current_target].append(value)
        elif kind == "linkmeta":
            parts = value.split("\t")
            if (not current_target or len(parts) != 3 or not all(parts) or
                    parts[2] not in {"imported", "local", "item"}):
                findings.append(Finding(snapshot, line_number, "closure-error",
                                        f"malformed link metadata `{value}`"))
                continue
            raw_link, resolved_target, authority = parts
            key = (current_target, raw_link)
            metadata = (resolved_target, authority)
            if key in link_metadata and link_metadata[key] != metadata:
                findings.append(Finding(snapshot, line_number, "closure-error",
                                        f"conflicting link metadata for `{raw_link}`"))
            link_metadata[key] = metadata
        else:
            findings.append(Finding(snapshot, line_number, "closure-error",
                                    f"unknown target-closure record `{kind}`"))

    if len(actual_targets) != len(set(actual_targets)):
        findings.append(Finding(snapshot, 1, "closure-error",
                                "target closure contains duplicate target records"))
    if set(actual_targets) != set(declared_targets):
        missing = sorted(set(declared_targets) - set(actual_targets))
        extra = sorted(set(actual_targets) - set(declared_targets))
        findings.append(Finding(snapshot, 1, "closure-error",
                                f"product target drift: missing={missing} extra={extra}"))
    if actual_sources != declared_sources:
        missing = sorted(declared_sources - actual_sources)
        extra = sorted(actual_sources - declared_sources)
        findings.append(Finding(snapshot, 1, "closure-error",
                                f"product source drift: missing={missing} extra={extra}"))
    linked_qualification = sorted(actual_sources & qualification)
    if linked_qualification:
        findings.append(Finding(snapshot, 1, "closure-error",
                                "qualification-only sources are product-linked: " +
                                ", ".join(linked_qualification)))

    final_link_keys: set[tuple[str, str]] = set()
    local_links: dict[str, list[str]] = {}
    for owner, target_links in links.items():
        for link in target_links:
            key = (owner, link)
            final_link_keys.add(key)
            metadata = link_metadata.get(key)
            if metadata is None:
                findings.append(Finding(snapshot, 1, "closure-error",
                                        f"final link has no marker metadata: `{link}`"))
                continue
            resolved_target, authority = metadata
            if authority == "local":
                if link not in declared_targets or resolved_target not in declared_targets:
                    findings.append(Finding(
                        snapshot, 1, "closure-error",
                        f"undeclared local target or alias `{link}` resolves to "
                        f"`{resolved_target}`",
                    ))
                else:
                    local_links.setdefault(owner, []).append(resolved_target)
            elif authority == "item":
                if (not link.startswith("-framework ") and
                        not Path(link).is_absolute()):
                    findings.append(Finding(snapshot, 1, "closure-error",
                                            f"unexpected linked item `{link}`"))

    for owner, raw_link in sorted(set(link_metadata) - final_link_keys):
        findings.append(Finding(snapshot, 1, "closure-error",
                                f"marker link removed from `{owner}`: `{raw_link}`"))

    reachable = {"laplace"}
    pending = ["laplace"]
    while pending:
        target = pending.pop()
        for link in local_links.get(target, []):
            if link not in reachable:
                reachable.add(link)
                pending.append(link)
    unreachable = sorted(set(declared_targets) - reachable)
    if unreachable:
        findings.append(Finding(snapshot, 1, "closure-error",
                                "declared product targets are not linked from laplace: " +
                                ", ".join(unreachable)))
    return findings


def _self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="laplace-universal-policy-") as raw:
        root = Path(raw)
        (root / "cmake").mkdir()
        (root / "src").mkdir()

        (root / "src" / "clean.cpp").write_text(
            """
#include "gguf_fixture.h"
#include <vector>
// ModelFamily::Qwen and Q4_K are historical words in a comment.
enum class Mode { Alpha, Beta };
bool choose(Mode mode) { return mode == Mode::Alpha; }
void report_failure() {
    fprintf(stderr, "unsupported source");
}
int generic_route(int arithmetic) { return arithmetic + 1; }
bool binding_matches(uint32_t artifact_id, uint32_t expected_artifact_id) {
    return artifact_id == expected_artifact_id;
}
""",
            encoding="utf-8",
        )
        (root / "src" / "qualification.cpp").write_text(
            "enum class ModelFamily { Qwen, Gemma };\n", encoding="utf-8"
        )
        (root / "src" / "gguf_fixture.h").write_text("", encoding="utf-8")
        catalog = root / "cmake" / "LaplaceProductSources.cmake"
        catalog.write_text(
            """
set(LAPLACE_PRODUCT_SOURCES
    src/clean.cpp
)
set(LAPLACE_QUALIFICATION_ONLY_SOURCES
    src/qualification.cpp
)
set(LAPLACE_PRODUCT_TARGETS laplace)
""",
            encoding="utf-8",
        )
        clean_findings = scan_catalog(catalog)
        if clean_findings:
            print("self-test: comments, standard includes, or unlisted sources were scanned")
            for finding in clean_findings:
                print(finding.render(root))
            return 1

        (root / "src" / "behavior.cpp").write_text(
            """
enum class ModelFamily { Qwen, Gemma };
enum class ExecutionShape { DenseLayer, MoeLayer };
bool select(std::string_view source_format, Digest artifact_hash,
            Digest golden_hash, uint32_t source_type_id,
            std::string_view source_tensor_name,
            const Operators& operators, Recipe recipe) {
    if (source_format == "Q4_K") return true;
    if (source_type_id == 12) return true;
    if (source_tensor_name == "blk.0.ffn_gate.weight") return true;
    if (artifact_hash == golden_hash) return true;
    if (operators.size() == 12 && operators[3].kind == OperatorKind::Rope) return true;
    switch (recipe.strategy) { case 7: return true; }
    return false;
}
""",
            encoding="utf-8",
        )
        catalog.write_text(
            "set(LAPLACE_PRODUCT_SOURCES src/behavior.cpp)\n"
            "set(LAPLACE_QUALIFICATION_ONLY_SOURCES)\n"
            "set(LAPLACE_PRODUCT_TARGETS laplace)\n",
            encoding="utf-8",
        )
        findings = scan_catalog(catalog)
        categories = {finding.category for finding in findings}
        expected = {
            "model-category",
            "renamed-category",
            "source-format-selector",
            "source-tensor-selector",
            "provenance-selector",
            "fixed-graph-matcher",
        }
        missing = expected - categories
        if missing:
            print("self-test: missing categories: " + ", ".join(sorted(missing)))
            for finding in findings:
                print(finding.render(root))
            return 1

        for variable, value in (
            ("LAPLACE_PRODUCT_SOURCES", "src/behavior.cpp"),
            ("LAPLACE_QUALIFICATION_ONLY_SOURCES", "src/qualification.cpp"),
            ("LAPLACE_PRODUCT_TARGETS", "hidden_product_library"),
        ):
            catalog.write_text(
                "set(LAPLACE_PRODUCT_SOURCES src/clean.cpp)\n"
                "set(LAPLACE_QUALIFICATION_ONLY_SOURCES)\n"
                "set(LAPLACE_PRODUCT_TARGETS laplace)\n"
                f"list(APPEND {variable} {value})\n",
                encoding="utf-8",
            )
            findings = scan_catalog(catalog)
            if not any(finding.category == "catalog-error" for finding in findings):
                print(f"self-test: unparsed mutation of {variable} was accepted")
                return 1

        catalog.write_text(
            "set(LAPLACE_PRODUCT_SOURCES src/clean.cpp)\n"
            "set(LAPLACE_QUALIFICATION_ONLY_SOURCES)\n"
            "set(LAPLACE_PRODUCT_TARGETS laplace)\n"
            "include(hidden_catalog_mutation.cmake)\n",
            encoding="utf-8",
        )
        if not any(finding.category == "catalog-error"
                   for finding in scan_catalog(catalog)):
            print("self-test: included catalog mutation was accepted")
            return 1

        (root / "src" / "include_root.cpp").write_text(
            "#define GENERATED_POLICY_FRAGMENT \"generated_policy.inc\"\n"
            "#include \"behavior.h\"\n"
            "#include \"behavior.inc\"\n"
            "#include GENERATED_POLICY_FRAGMENT\n"
            "#include <generated_policy.h>\n",
            encoding="utf-8",
        )
        (root / "src" / "behavior.h").write_text(
            "enum class ModelFamily { Qwen, Gemma };\n", encoding="utf-8"
        )
        (root / "src" / "behavior.inc").write_text(
            "enum class ExecutionShape { DenseLayer, MoeLayer };\n", encoding="utf-8"
        )
        (root / "src" / "generated_policy.inc").write_text(
            "constexpr unsigned source_type_id = 12;\n", encoding="utf-8"
        )
        catalog.write_text(
            "set(LAPLACE_PRODUCT_SOURCES src/include_root.cpp)\n"
            "set(LAPLACE_QUALIFICATION_ONLY_SOURCES)\n"
            "set(LAPLACE_PRODUCT_TARGETS laplace)\n",
            encoding="utf-8",
        )
        findings = scan_catalog(catalog)
        included = {finding.path.name for finding in findings}
        expected_includes = {"behavior.h", "behavior.inc", "generated_policy.inc"}
        if not expected_includes.issubset(included):
            print("self-test: repository-local include closure was not scanned: " +
                  ", ".join(sorted(expected_includes - included)))
            return 1
        if not any(finding.category == "include-error" and
                   "generated_policy.h" in finding.detail for finding in findings):
            print("self-test: unresolved generated angle include was accepted")
            return 1

        for label, source in (
            ("renamed variable",
             "auto failure_mode = \"Qwen\"; return dispatch(failure_mode);\n"),
            ("direct diagnostic string",
             "fprintf(stderr, \"Qwen\");\n"),
            ("diagnostic field flow",
             "policy.detail = \"Qwen\"; return dispatch(policy.detail);\n"),
            ("diagnostic return flow",
             "auto mode = failure(Code::X, \"Qwen\"); return dispatch(mode);\n"),
        ):
            (root / "src" / "semantic_bypass.cpp").write_text(source, encoding="utf-8")
            catalog.write_text(
                "set(LAPLACE_PRODUCT_SOURCES src/semantic_bypass.cpp)\n"
                "set(LAPLACE_QUALIFICATION_ONLY_SOURCES)\n"
                "set(LAPLACE_PRODUCT_TARGETS laplace)\n",
                encoding="utf-8",
            )
            if not any(finding.category == "source-format-selector"
                       for finding in scan_catalog(catalog)):
                print(f"self-test: {label} bypass was accepted")
                return 1

        for label, source in (
            ("sink definition",
             "#define fprintf route_by_source_name\n"
             "void f() { fprintf(stderr, \"Qwen\"); }\n"),
            ("sink alias",
             "#define emit_diagnostic fprintf\n"),
            ("sink undef",
             "#undef stderr\n"),
            ("sink conditional",
             "#if defined(fprintf)\n#endif\n"),
        ):
            (root / "src" / "semantic_bypass.cpp").write_text(source, encoding="utf-8")
            if not any(finding.category == "reserved-diagnostic-sink-macro"
                       for finding in scan_catalog(catalog)):
                print(f"self-test: {label} macro impersonation was accepted")
                return 1

        for label, source in (
            ("ordinary adjacent literals",
             "auto source = \"Q\" \"wen\"; return dispatch(source);\n"),
            ("comment-separated prefixed literals",
             "auto source = u8\"Q\" /* split */ u8\"wen\"; return dispatch(source);\n"),
            ("line-comment-separated literals",
             "auto source = \"Q\" // split\n \"wen\"; return dispatch(source);\n"),
            ("line-spliced literals",
             "auto source = \"Q\" \\\n \"wen\"; return dispatch(source);\n"),
            ("raw literals",
             "auto source = u8R\"(Q)\" u8R\"tag(wen)tag\"; return dispatch(source);\n"),
            ("escaped adjacent literals",
             "auto source = \"\\x51\" \"wen\"; return dispatch(source);\n"),
        ):
            (root / "src" / "semantic_bypass.cpp").write_text(source, encoding="utf-8")
            if not any(finding.category == "source-format-selector"
                       for finding in scan_catalog(catalog)):
                print(f"self-test: {label} were accepted")
                return 1

        (root / "src" / "library.cpp").write_text(
            "int library_value() { return 1; }\n", encoding="utf-8"
        )
        catalog.write_text(
            "set(LAPLACE_PRODUCT_SOURCES src/clean.cpp src/library.cpp)\n"
            "set(LAPLACE_QUALIFICATION_ONLY_SOURCES src/qualification.cpp)\n"
            "set(LAPLACE_PRODUCT_TARGETS laplace product_library)\n",
            encoding="utf-8",
        )
        snapshot = root / "target-closure.txt"
        snapshot.write_text(
            "target\tlaplace\n"
            "linkmeta\tproduct_library\tproduct_library\tlocal\n"
            "source\tsrc/clean.cpp\nlink\tproduct_library\n"
            "target\tproduct_library\nsource\tsrc/library.cpp\n",
            encoding="utf-8",
        )
        if verify_target_closure(catalog, snapshot):
            print("self-test: exact target closure was rejected")
            return 1
        snapshot.write_text(
            "target\tlaplace\n"
            "linkmeta\tproduct_library\tproduct_library\tlocal\n"
            "linkmeta\tExternal::Dependency\tExternal::Dependency\timported\n"
            "source\tsrc/clean.cpp\n"
            "link\tproduct_library\nlink\tExternal::Dependency\n"
            "target\tproduct_library\nsource\tsrc/library.cpp\n",
            encoding="utf-8",
        )
        if verify_target_closure(catalog, snapshot):
            print("self-test: imported target metadata was rejected")
            return 1
        for label, mutation in (
            ("source drift", "source\tsrc/behavior.cpp\n"),
            ("qualification link", "source\tsrc/qualification.cpp\n"),
            ("target drift", "link\thidden_product_library\n"),
        ):
            snapshot.write_text(
                "target\tlaplace\n"
                "linkmeta\tproduct_library\tproduct_library\tlocal\n"
                "source\tsrc/clean.cpp\nlink\tproduct_library\n" +
                mutation +
                "target\tproduct_library\nsource\tsrc/library.cpp\n",
                encoding="utf-8",
            )
            if not verify_target_closure(catalog, snapshot):
                print(f"self-test: {label} was accepted")
                return 1

        snapshot.write_text(
            "target\tlaplace\n"
            "linkmeta\tproduct_library\tproduct_library\tlocal\n"
            "source\tsrc/clean.cpp\n"
            "link\tproduct_library\nlink\tHidden::Product\n"
            "target\tproduct_library\nsource\tsrc/library.cpp\n",
            encoding="utf-8",
        )
        if not verify_target_closure(catalog, snapshot):
            print("self-test: post-marker namespaced link without metadata was accepted")
            return 1

        snapshot.write_text(
            "target\tlaplace\n"
            "linkmeta\tproduct_library\tproduct_library\tlocal\n"
            "linkmeta\tHidden::Product\thidden_product\tlocal\n"
            "source\tsrc/clean.cpp\n"
            "link\tproduct_library\nlink\tHidden::Product\n"
            "target\tproduct_library\nsource\tsrc/library.cpp\n",
            encoding="utf-8",
        )
        if not verify_target_closure(catalog, snapshot):
            print("self-test: namespaced local alias was accepted")
            return 1

    print("check_universal_product.py self-test: OK "
          "(CMake mutations/includes, repository/generated include closure, no diagnostic "
          "string exemption, reserved-sink macros, adjacent literals, diagnostic-flow bypasses, "
          "local-alias/target drift)")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return _self_test()
    if len(argv) == 3 and argv[0] == "--target-closure":
        catalog = Path(argv[1]).resolve()
        findings = verify_target_closure(catalog, Path(argv[2]).resolve())
        for finding in findings:
            print(finding.render(catalog.parent.parent))
        return 1 if findings else 0
    if len(argv) != 1:
        print("usage: check_universal_product.py [--self-test] "
              "[--target-closure <catalog> <snapshot>] <LaplaceProductSources.cmake>",
              file=sys.stderr)
        return 2
    catalog = Path(argv[0]).resolve()
    findings = scan_catalog(catalog)
    for finding in findings:
        print(finding.render(catalog.parent.parent))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
