#!/usr/bin/env python3
"""
AST Code Chunking, EmbeddingGemma Vectorization, 3D Dimensionality Reduction,
and Clustering Pipeline for the Official id Software DOOM Source Code.
"""

import json
import os
import re
import time
from collections import Counter, defaultdict
import numpy as np
import torch
import tree_sitter_c as tsc
from tree_sitter import Language, Parser
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import umap

C_LANGUAGE = Language(tsc.language())
parser = Parser(C_LANGUAGE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOOM_ROOT = os.path.join(BASE_DIR, "DOOM")
OUTPUT_JSON = os.path.join(BASE_DIR, "doom_code_space.json")
OUTPUT_NPZ = os.path.join(BASE_DIR, "doom_embeddings.npz")


def classify_subsystem(rel_path: str) -> str:
    parts = rel_path.replace("\\", "/").split("/")
    folder = parts[0] if len(parts) > 1 else ""
    fname = parts[-1].lower()

    if folder in ("ipx", "sersrc"):
        return "DOS Network & Serial (ipx/sersrc)"
    if folder == "sndserv" or fname.startswith(("s_sound", "sounds.")):
        return "Sound & Audio Server (S_*/sndserv)"
    if fname.startswith("p_"):
        return "Playsim, AI & World Specials (P_*)"
    if fname.startswith(("r_", "tables.")):
        return "3D BSP Software Renderer (R_*)"
    if fname.startswith(("am_", "hu_", "st_", "wi_", "f_")):
        return "HUD, Automap, Status Bar & Finale (AM/HU/ST/WI/F)"
    if fname.startswith("m_"):
        return "Menus, Fixed-Point Math & Utils (M_*)"
    if fname.startswith("i_"):
        return "OS, X11 Video & Hardware Interface (I_*)"
    if fname.startswith(("w_", "z_", "v_")):
        return "Engine Core: WAD, Zone Memory & Video (W/Z/V)"
    return "Game Loop, Net Ticcmds & Entity Tables (D/G/info)"


def get_node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("latin-1", errors="replace")


def compute_ast_metrics(node, code_bytes: bytes):
    """Compute AST node count, max depth, cyclomatic complexity, and called functions."""
    node_count = 0
    max_depth = 0
    complexity = 1
    calls = []

    branch_types = {
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "case_statement",
        "conditional_expression",
    }

    stack = [(node, 1)]
    while stack:
        curr, depth = stack.pop()
        node_count += 1
        if depth > max_depth:
            max_depth = depth

        ctype = curr.type
        if ctype in branch_types:
            complexity += 1
        elif ctype == "binary_expression":
            for ch in curr.children:
                if ch.type in ("&&", "||"):
                    complexity += 1
        elif ctype == "call_expression":
            fn_node = curr.child_by_field_name("function")
            if fn_node is not None and fn_node.type == "identifier":
                fn_name = get_node_text(code_bytes, fn_node).strip()
                if fn_name and fn_name not in calls:
                    calls.append(fn_name)

        for ch in reversed(curr.children):
            stack.append((ch, depth + 1))

    return node_count, max_depth, complexity, calls


def extract_function_name_and_signature(node, code_bytes: bytes):
    """Extract function identifier and clean signature from a function_definition AST node."""
    declarator = node.child_by_field_name("declarator")
    func_name = None

    curr = declarator
    while curr is not None:
        if curr.type == "identifier":
            func_name = get_node_text(code_bytes, curr).strip()
            break
        elif curr.type == "function_declarator":
            curr = curr.child_by_field_name("declarator")
        elif curr.type in ("pointer_declarator", "parenthesized_declarator"):
            curr = curr.child_by_field_name("declarator") or (curr.children[1] if len(curr.children) > 1 else None)
        else:
            found = None
            for ch in curr.children:
                if ch.type in ("function_declarator", "pointer_declarator", "identifier"):
                    found = ch
                    break
            curr = found

    if not func_name:
        func_name = "anonymous_fn"

    body = node.child_by_field_name("body")
    if body is not None:
        sig_bytes = code_bytes[node.start_byte:body.start_byte]
        sig = re.sub(r"\s+", " ", sig_bytes.decode("latin-1", errors="replace")).strip()
    else:
        sig = func_name
    return func_name, sig


def extract_type_or_decl_name(node, code_bytes: bytes):
    """Extract the primary identifier from a type_definition, struct_specifier, enum_specifier, or declaration."""
    ntype = node.type

    if ntype == "type_definition":
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            stack = [decl]
            while stack:
                curr = stack.pop()
                if curr.type in ("type_identifier", "identifier"):
                    return get_node_text(code_bytes, curr).strip()
                stack.extend(reversed(curr.children))
        for ch in reversed(node.children):
            if ch.type in ("type_identifier", "identifier"):
                return get_node_text(code_bytes, ch).strip()

    if ntype in ("struct_specifier", "enum_specifier", "union_specifier"):
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return f"{ntype.split('_')[0]} {get_node_text(code_bytes, name_node).strip()}"

    if ntype == "declaration":
        for ch in node.children:
            if ch.type in ("struct_specifier", "enum_specifier", "union_specifier"):
                name_node = ch.child_by_field_name("name")
                if name_node is not None:
                    return f"{ch.type.split('_')[0]} {get_node_text(code_bytes, name_node).strip()}"
            if ch.type == "init_declarator":
                decl_node = ch.child_by_field_name("declarator")
                if decl_node is not None:
                    stack = [decl_node]
                    while stack:
                        curr = stack.pop()
                        if curr.type == "identifier":
                            return get_node_text(code_bytes, curr).strip()
                        stack.extend(reversed(curr.children))
            if ch.type in ("identifier", "array_declarator", "pointer_declarator"):
                stack = [ch]
                while stack:
                    curr = stack.pop()
                    if curr.type == "identifier":
                        return get_node_text(code_bytes, curr).strip()
                    stack.extend(reversed(curr.children))

    first_line = get_node_text(code_bytes, node).splitlines()[0].strip()
    return first_line[:48]


def is_banner_comment(text: str) -> bool:
    lower = text.lower()
    if "doom source code license" in lower or "emacs style mode select" in lower:
        return True
    if "$id:" in lower or "$log:" in lower:
        return True
    return False


def extract_file_description(root_node, code_bytes: bytes) -> str:
    for ch in root_node.children[:15]:
        if ch.type == "comment":
            txt = get_node_text(code_bytes, ch)
            m = re.search(r"DESCRIPTION:\s*(.*?)(?:\n\s*//---|\*/|$)", txt, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
                lines = [re.sub(r"^\s*(//|\*)\s?", "", ln).strip() for ln in raw.splitlines()]
                cleaned = " ".join([ln for ln in lines if ln and not ln.startswith("---")])
                if cleaned:
                    return cleaned
    return ""


def parse_all_doom_files():
    chunks = []
    all_files = []
    for root, _, fnames in os.walk(DOOM_ROOT):
        for f in sorted(fnames):
            if f.endswith((".c", ".h", ".C", ".H")):
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, DOOM_ROOT).replace("\\", "/")
                all_files.append((rel_p, full_p))

    all_files.sort(key=lambda x: x[0])

    for rel_path, full_path in all_files:
        with open(full_path, "rb") as f:
            code_bytes = f.read()

        tree = parser.parse(code_bytes)
        subsystem = classify_subsystem(rel_path)
        file_desc = extract_file_description(tree.root_node, code_bytes)
        filename = os.path.basename(rel_path)

        flat_nodes = []

        def collect_nodes(parent):
            for ch in parent.children:
                if ch.type in ("preproc_ifdef", "preproc_ifndef", "preproc_if", "preproc_else", "preproc_elif"):
                    collect_nodes(ch)
                else:
                    flat_nodes.append(ch)

        collect_nodes(tree.root_node)

        pending_comments = []
        last_comment_end_line = -10
        leftover_nodes = []
        file_major_chunks = 0

        for node in flat_nodes:
            ntype = node.type
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            line_span = end_line - start_line + 1

            if ntype == "comment":
                ctxt = get_node_text(code_bytes, node).strip()
                if not is_banner_comment(ctxt):
                    if start_line - last_comment_end_line <= 2:
                        pending_comments.append(ctxt)
                    else:
                        pending_comments = [ctxt]
                    last_comment_end_line = end_line
                continue

            is_primary = False
            ast_label = ntype
            category = "Other"

            if ntype == "function_definition":
                is_primary = True
                ast_label = "function_definition"
                category = "Function"
            elif ntype == "type_definition":
                is_primary = True
                ast_label = "type_definition"
                category = "Struct / Typedef"
            elif ntype in ("struct_specifier", "union_specifier"):
                is_primary = True
                ast_label = ntype
                category = "Struct / Typedef"
            elif ntype == "enum_specifier":
                is_primary = True
                ast_label = "enum_specifier"
                category = "Enum"
            elif ntype == "declaration":
                subtypes = [c.type for c in node.children]
                if "struct_specifier" in subtypes or "union_specifier" in subtypes:
                    spec = next(c for c in node.children if c.type in ("struct_specifier", "union_specifier"))
                    if any(sc.type == "field_declaration_list" for sc in spec.children):
                        is_primary = True
                        ast_label = "struct_specifier"
                        category = "Struct / Typedef"
                elif "enum_specifier" in subtypes:
                    spec = next(c for c in node.children if c.type == "enum_specifier")
                    if any(sc.type == "enumerator_list" for sc in spec.children):
                        is_primary = True
                        ast_label = "enum_specifier"
                        category = "Enum"
                elif any(c.type == "init_declarator" for c in node.children) and line_span >= 5:
                    txt = get_node_text(code_bytes, node)
                    if "rcsid" not in txt:
                        is_primary = True
                        ast_label = "declaration (data_table)"
                        category = "Data Table"

            if is_primary:
                doc_comment = ""
                if pending_comments and (start_line - last_comment_end_line <= 4):
                    doc_comment = "\n".join(pending_comments)
                pending_comments = []

                if ntype == "function_definition":
                    name, signature = extract_function_name_and_signature(node, code_bytes)
                else:
                    name = extract_type_or_decl_name(node, code_bytes)
                    first_ln = get_node_text(code_bytes, node).splitlines()[0].strip()
                    signature = first_ln[:120]

                node_count, max_depth, complexity, calls = compute_ast_metrics(node, code_bytes)
                raw_code = get_node_text(code_bytes, node)
                code_lines = raw_code.splitlines()

                if len(code_lines) > 130:
                    display_code = "\n".join(code_lines[:105]) + f"\n\n/* ... [{len(code_lines) - 120} lines omitted in preview] ... */\n\n" + "\n".join(code_lines[-15:])
                else:
                    display_code = raw_code

                embed_code = "\n".join(code_lines[:45])[:1300]

                chunks.append({
                    "name": name,
                    "ast_type": ast_label,
                    "category": category,
                    "file": rel_path,
                    "filename": filename,
                    "subsystem": subsystem,
                    "file_description": file_desc,
                    "start_line": start_line,
                    "end_line": end_line,
                    "line_count": line_span,
                    "ast_node_count": node_count,
                    "ast_depth": max_depth,
                    "cyclomatic_complexity": complexity,
                    "calls": calls[:30],
                    "signature": signature,
                    "doc_comment": doc_comment,
                    "code": display_code,
                    "embed_code": embed_code,
                })
                file_major_chunks += 1
            else:
                if ntype in ("preproc_def", "preproc_function_def", "declaration"):
                    txt = get_node_text(code_bytes, node).strip()
                    if "rcsid" not in txt and not txt.startswith("#ifndef __") and not txt.startswith("#define __"):
                        leftover_nodes.append(node)
                pending_comments = []

        if leftover_nodes and (file_major_chunks == 0 or len(leftover_nodes) >= 8):
            first_n = leftover_nodes[0]
            last_n = leftover_nodes[-1]
            start_line = first_n.start_point[0] + 1
            end_line = last_n.end_point[0] + 1
            combined_lines = []
            total_ast_nodes = 0
            max_d = 0
            defined_macros = []
            for ln_node in leftover_nodes:
                nc, md, _, _ = compute_ast_metrics(ln_node, code_bytes)
                total_ast_nodes += nc
                if md > max_d:
                    max_d = md
                ntxt = get_node_text(code_bytes, ln_node).strip()
                combined_lines.append(ntxt)
                if ln_node.type in ("preproc_def", "preproc_function_def"):
                    name_n = ln_node.child_by_field_name("name")
                    if name_n:
                        defined_macros.append(get_node_text(code_bytes, name_n).strip())
                elif ln_node.type == "declaration":
                    dname = extract_type_or_decl_name(ln_node, code_bytes)
                    if dname:
                        defined_macros.append(dname)

            raw_combined = "\n".join(combined_lines)
            c_lines = raw_combined.splitlines()
            if len(c_lines) > 120:
                display_code = "\n".join(c_lines[:100]) + f"\n\n/* ... [{len(c_lines) - 115} lines omitted] ... */\n\n" + "\n".join(c_lines[-15:])
            else:
                display_code = raw_combined

            chunk_title = f"{filename} [Macros & Decls]"
            sig_preview = ", ".join(defined_macros[:8])
            if len(defined_macros) > 8:
                sig_preview += f" (+{len(defined_macros) - 8} more)"

            chunks.append({
                "name": chunk_title,
                "ast_type": "preproc_def / declaration",
                "category": "Header / Module Declarations",
                "file": rel_path,
                "filename": filename,
                "subsystem": subsystem,
                "file_description": file_desc,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": len(c_lines),
                "ast_node_count": total_ast_nodes,
                "ast_depth": max_d,
                "cyclomatic_complexity": 1,
                "calls": [],
                "signature": sig_preview or chunk_title,
                "doc_comment": file_desc,
                "code": display_code,
                "embed_code": "\n".join(c_lines[:40])[:1200],
            })

    name_to_id = {}
    for idx, ch in enumerate(chunks):
        ch["id"] = idx
        if ch["name"] not in name_to_id or ch["ast_type"] == "function_definition":
            name_to_id[ch["name"]] = idx

    for ch in chunks:
        resolved_callees = []
        for callee_name in ch["calls"]:
            if callee_name in name_to_id and name_to_id[callee_name] != ch["id"]:
                resolved_callees.append(name_to_id[callee_name])
        ch["callee_ids"] = resolved_callees

    caller_map = defaultdict(list)
    for ch in chunks:
        for cid in ch["callee_ids"]:
            if ch["id"] not in caller_map[cid]:
                caller_map[cid].append(ch["id"])
    for ch in chunks:
        ch["caller_ids"] = caller_map[ch["id"]][:35]

    return chunks


def format_for_embeddinggemma(ch: dict) -> str:
    """Format an AST chunk using EmbeddingGemma's clustering prompt and rich AST metadata."""
    calls_str = ", ".join(ch["calls"][:15]) if ch["calls"] else "none"
    comment_clean = re.sub(r"//+|\*+|/+", " ", ch["doc_comment"] or "").strip()
    comment_clean = re.sub(r"\s+", " ", comment_clean)[:300]

    header = (
        f"task: clustering | query: "
        f"DOOM C AST Chunk | Subsystem: {ch['subsystem']} | File: {ch['file']} ({ch['file_description']}) | "
        f"AST Node: {ch['ast_type']} ({ch['category']}) | Symbol: {ch['name']} | "
        f"Signature: {ch['signature']} | Calls: {calls_str}"
    )
    if comment_clean:
        header += f" | Doc: {comment_clean}"
    return f"{header}\n```c\n{ch['embed_code']}\n```"


def scale_coords_3d(coords: np.ndarray, target_range: float = 90.0) -> np.ndarray:
    """Center and scale 3D coordinates into [-target_range, target_range]."""
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    max_abs = np.percentile(np.abs(centered), 98)
    if max_abs < 1e-6:
        max_abs = 1.0
    scaled = (centered / max_abs) * target_range
    return np.clip(scaled, -target_range * 1.35, target_range * 1.35)


def split_identifier_tokens(text: str) -> str:
    """Split camelCase and snake_case C identifiers into readable tokens for TF-IDF cluster naming."""
    # Remove common 1-2 letter DOOM prefixes like P_, R_, M_, ST_, HU_, WI_, AM_, I_, S_, W_, Z_, V_, G_, D_, A_
    cleaned = re.sub(r"\b(P|R|M|ST|HU|WI|AM|I|S|W|Z|V|G|D|A|F)_", " ", text)
    # Split CamelCase
    cleaned = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", cleaned)
    return cleaned.lower()


def generate_cluster_summaries(chunks, embeddings, coords_umap, labels, n_clusters):
    """Generate rich semantic cluster names, top keywords, centroids, and exemplar AST nodes."""
    cluster_docs = []
    cluster_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        cluster_indices[int(lbl)].append(idx)

    stop_words = {
        "void", "int", "char", "short", "long", "unsigned", "signed", "const", "static",
        "extern", "struct", "enum", "typedef", "boolean", "true", "false", "null", "return",
        "if", "else", "for", "while", "switch", "case", "break", "default", "sizeof",
        "fixed", "byte", "decls", "macros", "doom", "linuxdoom", "function", "definition",
        "type", "declaration", "file", "data", "table", "value", "index", "count", "num",
    }

    for cid in range(n_clusters):
        idxs = cluster_indices[cid]
        doc_parts = []
        for i in idxs:
            ch = chunks[i]
            doc_parts.append(split_identifier_tokens(ch["name"]) * 3)
            doc_parts.append(split_identifier_tokens(ch["doc_comment"]))
            doc_parts.append(split_identifier_tokens(ch["file_description"]))
            doc_parts.append(split_identifier_tokens(" ".join(ch["calls"][:8])))
        cluster_docs.append(" ".join(doc_parts))

    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        max_features=1000,
    )
    tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    feature_names = np.array(vectorizer.get_feature_names_out())

    clusters_meta = []
    for cid in range(n_clusters):
        idxs = cluster_indices[cid]
        row = tfidf_matrix[cid].toarray().flatten()
        top_term_indices = row.argsort()[::-1][:8]
        top_terms = [str(feature_names[ti]) for ti in top_term_indices if row[ti] > 0][:6]

        # Dominant subsystem & file prefixes
        sub_counts = Counter(chunks[i]["subsystem"] for i in idxs)
        dom_subsystem = sub_counts.most_common(1)[0][0]
        short_sub = dom_subsystem.split("(")[0].strip()

        cat_counts = Counter(chunks[i]["category"] for i in idxs)
        dom_cat = cat_counts.most_common(1)[0][0]

        # 3D centroid in UMAP space
        c_umap = np.mean(coords_umap[idxs], axis=0)
        # 768D centroid for finding exemplar AST nodes
        c_emb = np.mean(embeddings[idxs], axis=0)
        c_emb = c_emb / (np.linalg.norm(c_emb) + 1e-9)
        sims = embeddings[idxs] @ c_emb
        top_local = np.argsort(sims)[::-1][:5]
        exemplars = [chunks[idxs[li]]["name"] for li in top_local]

        kw_title = ", ".join(t.capitalize() for t in top_terms[:3]) if top_terms else short_sub
        label_name = f"C{cid:02d}: {kw_title} ({short_sub})"

        clusters_meta.append({
            "id": cid,
            "name": label_name,
            "short_name": f"C{cid:02d}: {kw_title}",
            "count": len(idxs),
            "keywords": top_terms,
            "dominant_subsystem": dom_subsystem,
            "dominant_category": dom_cat,
            "centroid_umap": [round(float(x), 3) for x in c_umap],
            "exemplars": exemplars,
        })

    return clusters_meta


def main():
    t_start = time.time()
    print("[1/5] Parsing DOOM C/H files via Tree-sitter C AST...")
    chunks = parse_all_doom_files()
    print(f"      Extracted {len(chunks)} AST chunks from {len(set(c['file'] for c in chunks))} files.")

    print("[2/5] Loading google/embeddinggemma-2 and encoding AST chunks...")
    model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None, "audio_config": None})
    model.max_seq_length = 384

    texts = [format_for_embeddinggemma(c) for c in chunks]
    t_emb = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"      Encoded {embeddings.shape[0]} chunks into {embeddings.shape[1]}D vectors in {time.time() - t_emb:.1f}s.")

    print("[3/5] Computing exact Top-6 Cosine Nearest Neighbors in 768D EmbeddingGemma space...")
    sim_matrix = embeddings @ embeddings.T
    np.fill_diagonal(sim_matrix, -1.0)
    for i, ch in enumerate(chunks):
        top_k_idx = np.argsort(sim_matrix[i])[::-1][:6]
        ch["neighbors"] = [
            {"id": int( idx ), "sim": round(float(sim_matrix[i, idx]), 4)}
            for idx in top_k_idx
        ]
        # Remove temporary embed_code field before JSON export
        ch.pop("embed_code", None)

    print("[4/5] Computing 3D Projections (UMAP, t-SNE, PCA) from 768D EmbeddingGemma space...")
    reducer_umap = umap.UMAP(
        n_components=3,
        n_neighbors=18,
        min_dist=0.12,
        metric="cosine",
        random_state=42,
    )
    coords_umap = scale_coords_3d(reducer_umap.fit_transform(embeddings))

    pca_3d = PCA(n_components=3, random_state=42)
    coords_pca = scale_coords_3d(pca_3d.fit_transform(embeddings))
    pca_var = float(np.sum(pca_3d.explained_variance_ratio_)) * 100.0

    tsne_3d = TSNE(
        n_components=3,
        perplexity=28,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    coords_tsne = scale_coords_3d(tsne_3d.fit_transform(embeddings))

    print("[5/5] Clustering AST chunks in EmbeddingGemma space...")
    # Combine L2-normalized 768D embeddings with normalized 3D UMAP structure for semantic + manifold clustering
    umap_unit = coords_umap / (np.linalg.norm(coords_umap, axis=1, keepdims=True) + 1e-9)
    joint_features = np.hstack([embeddings * 0.75, umap_unit * 0.25])

    n_clusters = 12
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(joint_features)
    sil_umap = float(silhouette_score(coords_umap, cluster_labels))
    sil_high = float(silhouette_score(embeddings, cluster_labels, metric="cosine"))

    # Also run HDBSCAN on the 3D UMAP space
    hdb = HDBSCAN(min_cluster_size=14, min_samples=5)
    hdb_labels = hdb.fit_predict(coords_umap)
    n_hdb_clusters = len(set(hdb_labels) - {-1})

    print(f"      K-Means (K={n_clusters}) Silhouette (3D UMAP): {sil_umap:.3f}, (768D Cosine): {sil_high:.3f}")
    print(f"      HDBSCAN discovered {n_hdb_clusters} density clusters.")

    clusters_meta = generate_cluster_summaries(chunks, embeddings, coords_umap, cluster_labels, n_clusters)

    for i, ch in enumerate(chunks):
        ch["x"] = round(float(coords_umap[i, 0]), 3)
        ch["y"] = round(float(coords_umap[i, 1]), 3)
        ch["z"] = round(float(coords_umap[i, 2]), 3)
        ch["tsne_x"] = round(float(coords_tsne[i, 0]), 3)
        ch["tsne_y"] = round(float(coords_tsne[i, 1]), 3)
        ch["tsne_z"] = round(float(coords_tsne[i, 2]), 3)
        ch["pca_x"] = round(float(coords_pca[i, 0]), 3)
        ch["pca_y"] = round(float(coords_pca[i, 1]), 3)
        ch["pca_z"] = round(float(coords_pca[i, 2]), 3)
        ch["cluster"] = int(cluster_labels[i])
        ch["cluster_name"] = clusters_meta[int(cluster_labels[i])]["short_name"]
        ch["hdbscan_cluster"] = int(hdb_labels[i])

    # Compute additional 3D centroids for t-SNE and PCA per cluster
    for cmeta in clusters_meta:
        cid = cmeta["id"]
        idxs = [i for i, lbl in enumerate(cluster_labels) if int(lbl) == cid]
        c_tsne = np.mean(coords_tsne[idxs], axis=0)
        c_pca = np.mean(coords_pca[idxs], axis=0)
        cmeta["centroid_tsne"] = [round(float(x), 3) for x in c_tsne]
        cmeta["centroid_pca"] = [round(float(x), 3) for x in c_pca]

    payload = {
        "metadata": {
            "repo": "https://github.com/id-Software/DOOM.git",
            "model": "google/embeddinggemma-2",
            "embedding_dim": int(embeddings.shape[1]),
            "ast_parser": "tree-sitter-c (0.24.2)",
            "total_chunks": len(chunks),
            "total_files": len(set(c["file"] for c in chunks)),
            "total_call_edges": sum(len(c["callee_ids"]) for c in chunks),
            "n_clusters": n_clusters,
            "n_hdbscan_clusters": n_hdb_clusters,
            "silhouette_3d_umap": round(sil_umap, 4),
            "silhouette_768d_cosine": round(sil_high, 4),
            "pca_3d_variance_pct": round(pca_var, 2),
        },
        "clusters": clusters_meta,
        "chunks": chunks,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        coords_umap=coords_umap,
        coords_tsne=coords_tsne,
        coords_pca=coords_pca,
        cluster_labels=cluster_labels,
        hdb_labels=hdb_labels,
    )

    print(f"Done in {time.time() - t_start:.1f}s! Saved {OUTPUT_JSON} and {OUTPUT_NPZ}.")
    for cmeta in clusters_meta:
        print(f"  [{cmeta['short_name']}] ({cmeta['count']} chunks) -> Exemplars: {', '.join(cmeta['exemplars'][:3])}")


if __name__ == "__main__":
    main()
