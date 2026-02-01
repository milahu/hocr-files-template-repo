#!/usr/bin/env python3

import os
import sys
import zlib
import zipfile
import inspect
import hashlib
import textwrap
import subprocess
from pathlib import Path


def fatal(msg):
    print(f"fatal: {msg}", file=sys.stderr)
    sys.exit(1)


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def write_object(obj_type, data):
    """
    Write a Git object to .git/objects and return its hash.
    `data` is the raw content of the object (tree/blob/commit).
    """
    header = f"{obj_type} {len(data)}\0".encode()
    full = header + data
    digest = hashlib.sha1(full).hexdigest()  # SHA-1 over "tree <len>\0<content>"

    # store in .git/objects (optional, only if reconstructing)
    obj_dir = os.path.join(".git", "objects", digest[:2])
    obj_path = os.path.join(obj_dir, digest[2:])
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    if not os.path.exists(obj_path):
        with open(obj_path, "wb") as f:
            f.write(zlib.compress(full))

    return digest


def read_commit_object():
    commits = [f for f in os.listdir(".") if f.startswith(MAGIC_PREFIX)]
    if len(commits) != 1:
        fatal("expected exactly one embedded commit object")

    path = commits[0]
    commit_hash = path[len(MAGIC_PREFIX):-4]

    with open(path, "rb") as f:
        commit_body = f.read()

    lines = commit_body.splitlines()
    tree_line = next((l for l in lines if l.startswith(b"tree ")), None)
    if not tree_line:
        fatal("commit has no tree")

    tree_hash = tree_line.split()[1].decode()
    return commit_hash, tree_hash, commit_body


def iter_working_files():
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            # skip magic/embedded files
            if name == MAGIC_SCRIPT:
                continue
            if name.startswith(MAGIC_PREFIX) and name.endswith(".txt"):
                continue
            if name.startswith(".git-tree-") and name.endswith(".txt"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, ".")
            yield rel


def file_mode(path):
    st = os.lstat(path)
    if os.path.islink(path):
        return "120000"
    if st.st_mode & 0o111:
        return "100755"
    return "100644"


def read_file(path):
    if os.path.islink(path):
        return os.readlink(path).encode()
    with open(path, "rb") as f:
        return f.read()


def build_blobs():
    blobs = {}
    for path in iter_working_files():
        data = read_file(path)
        h = write_object("blob", data)
        blobs[path] = h
    return blobs


def build_trees(blobs, debug_subtree=None):
    """
    Build all Git tree objects from blobs.
    If debug_subtree is set (e.g., "OEBPS"), dump the serialized tree content to file for inspection.

    Returns:
        root_tree_hash (str): SHA-1 of the root tree
        trees (dict): mapping from directory path -> tree hash
    """
    # debug
    # debug_subtree = "OEBPS" # subtree
    # debug_subtree = "" # root tree

    trees = {}  # directory path -> tree hash

    # collect all directories
    dirs = set()
    for path in blobs:
        d = os.path.dirname(path)
        while True:
            # if d == ".":
            #     d = ""
            dirs.add(d)
            if d == "":
                break
            d = os.path.dirname(d)

    # assert "" in dirs, "root directory missing"
    # assert "." not in dirs, "unexpected '.' directory"

    # build trees from deepest first
    # for d in sorted(dirs, key=lambda x: x.count(os.sep), reverse=True):
    for d in sorted(dirs, key=lambda x: (x.count(os.sep), x), reverse=True):
        entries = []

        # blobs in this directory
        for path, blob_hash in blobs.items():
            if os.path.dirname(path) == d:
                name_bytes = os.path.basename(path).encode()
                mode_bytes = file_mode(path).encode()
                sha_bytes = bytes.fromhex(blob_hash)
                entries.append((name_bytes, mode_bytes, sha_bytes))

        # subtrees in this directory
        for subdir in dirs:
            if subdir == "" or os.path.dirname(subdir) != d:
                continue

            if subdir not in trees:
                # subtree not built yet (should only happen transiently)
                continue

            tree_hash = trees[subdir]

            name_bytes = os.path.basename(subdir).encode()
            mode_bytes = b"40000"
            sha_bytes = bytes.fromhex(tree_hash)

            entries.append((name_bytes, mode_bytes, sha_bytes))

        # sort entries by name bytes
        # entries.sort(key=lambda e: e[0])
        def git_tree_sort_key(entry):
            name = entry[0]
            mode = entry[1]
            # if name == b'OEBPS':
            #     print("build_trees: entry", repr(mode), repr(name))
            if mode == b'40000' or mode == b'040000':
                # print("build_trees: entry", repr(mode), repr(name))
                return name + b"/"
            return name
        entries.sort(key=git_tree_sort_key)

        # serialize tree object
        body = b"".join(mode + b" " + name + b"\0" + sha for name, mode, sha in entries)

        # write object
        tree_hash = write_object("tree", body)
        trees[d] = tree_hash

        # DEBUG: dump OEBPS subtree bytes
        if debug_subtree is not None and d == debug_subtree:
            dump_path = f"tree-debug-{debug_subtree}.bin"
            with open(dump_path, "wb") as f:
                f.write(body)
            print(f"DEBUG: {debug_subtree} tree serialized to {dump_path}")
            print(f"DEBUG: {debug_subtree} tree hash: {tree_hash}")

    root_tree_hash = trees[""]
    return root_tree_hash, trees


def dump_tree_text(blobs, trees):
    """
    Produce Git-style tree dump, exactly like:
    git ls-tree -r -t --full-tree HEAD
    Output is bytes, sorted by full path.
    """
    entries = []

    # add trees
    for path, tree_hash in trees.items():
        if path == "":
            continue
        entries.append((path.encode(), b"040000", b"tree", tree_hash.encode()))
        # entries.append((path.encode(), b"40000", b"tree", tree_hash.encode()))

    # add blobs
    for path, blob_hash in blobs.items():
        mode = file_mode(path).encode()
        entries.append((path.encode(), mode, b"blob", blob_hash.encode()))

    # sort by path bytes
    # entries.sort(key=lambda e: e[0])
    def git_tree_sort_key(entry):
        name = entry[0]
        mode = entry[1]
        # if name == b'OEBPS':
        #     print("dump_tree_text: entry", repr(mode), repr(name))
        if mode == b'40000' or mode == b'040000':
            # print("dump_tree_text: entry", repr(mode), repr(name))
            return name + b"/"
        return name
    entries.sort(key=git_tree_sort_key)

    # render lines
    lines = [b"%s %s %s\t%s" % (mode, typ, h, path) for path, mode, typ, h in entries]

    return b"\n".join(lines) + b"\n"


def git_sort_key(name_bytes):
    """
    Sort tree entries like git:
    . < 0-9 < A-Z < a-z < other bytes
    """
    order = []
    for b in name_bytes:
        # ASCII ranges
        if b == 0x2E:        # '.'
            order.append(0)
        elif 0x30 <= b <= 0x39:  # '0'-'9'
            order.append(1 << 8 | b)
        elif 0x41 <= b <= 0x5A:  # 'A'-'Z'
            order.append(2 << 8 | b)
        elif 0x61 <= b <= 0x7A:  # 'a'-'z'
            order.append(3 << 8 | b)
        else:                  # all others
            order.append(4 << 8 | b)
    return tuple(order)


def read_expected_tree(tree_hash):
    path = f".git-tree-{tree_hash}.txt"
    if not os.path.exists(path):
        fatal("missing expected tree file " + path)

    with open(path, "rb") as f:
        return f.read()


def git_init_main():
    if os.path.exists(".git"):
        fatal(".git already exists")

    os.makedirs(".git/objects", exist_ok=True)
    os.makedirs(".git/refs/heads", exist_ok=True)

    commit_hash, expected_tree, commit_body = read_commit_object()

    # write commit object verbatim
    actual_commit = write_object("commit", commit_body)
    if actual_commit != commit_hash:
        fatal("commit hash mismatch")

    blobs = build_blobs()
    root_tree, trees = build_trees(blobs)

    if root_tree != expected_tree:
        expected_txt = read_expected_tree(expected_tree)
        actual_txt = dump_tree_text(blobs, trees)

        with open(".git-tree-actual.txt", "wb") as f:
            f.write(actual_txt)

        fatal(
            "reconstructed tree does not match commit\n\n"
            "expected tree hash: " + expected_tree + "\n"
            "actual tree hash:   " + root_tree + "\n\n"
            "wrote reconstructed tree to .git-tree-actual.txt\n"
            "diff with:\n"
            f"  diff -u .git-tree-{expected_tree}.txt .git-tree-actual.txt"
        )

    with open(".git/HEAD", "w") as f:
        f.write("ref: refs/heads/main\n")

    with open(".git/refs/heads/main", "w") as f:
        f.write(commit_hash + "\n")

    with open(".git/shallow", "w") as f:
        f.write(commit_hash + "\n")

    subprocess.check_call(["git", "read-tree", "HEAD"])

    # remove untracked files
    os.unlink(".git-init.py")
    os.unlink(f".git-tree-{root_tree}.txt")
    os.unlink(f".git-commit-{commit_hash}.txt")

    subprocess.check_call(["git", "status"])

    print("Initialized shallow git repository at", commit_hash)


GIT_INIT_FUNCTIONS = [
    fatal,
    sha1,
    write_object,
    read_commit_object,
    iter_working_files,
    file_mode,
    read_file,
    build_blobs,
    build_trees,
    dump_tree_text,
    git_sort_key,
    read_expected_tree,
    git_init_main,
]


def generate_git_init_source():
    parts = []

    # shebang + imports
    parts.append(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "import hashlib\n"
        "import zlib\n\n"
        "import subprocess\n\n"
        "MAGIC_PREFIX = '.git-commit-'\n"
        "MAGIC_SCRIPT = '.git-init.py'\n\n"
    )

    for fn in GIT_INIT_FUNCTIONS:
        src = inspect.getsource(fn)
        parts.append(textwrap.dedent(src))
        parts.append("\n\n")

    parts.append(
        "if __name__ == '__main__':\n"
        "    git_init_main()\n"
    )

    return "".join(parts)


def git(cmd, input=None, text=False):
    return subprocess.check_output(
        ["git"] + cmd,
        input=input,
        text=text,
    )


def fatal(msg):
    print(f"fatal: {msg}", file=sys.stderr)
    sys.exit(1)


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def zipinfo(name, mode=0o644, compress=zipfile.ZIP_DEFLATED):
    zi = zipfile.ZipInfo(name)
    zi.date_time = ZIP_EPOCH
    zi.compress_type = compress
    zi.external_attr = (mode & 0o777) << 16
    zi.create_system = 3
    zi.extra = b""
    return zi


def main(out_epub):
    # Ensure we're in a git repo
    try:
        commit_hash = git(["rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        fatal("not a git repository or HEAD not found")

    commit_date = git(["show", "-s", "--format=%cI", "HEAD"], text=True).strip()
    commit_date = commit_date[:10]

    if out_epub is None:
        # short_commit_hash_len = 7 # github
        short_commit_hash_len = 10 # gitea
        out_epub = (
            os.path.basename(os.path.abspath(".")) +
            "." +
            commit_date +
            "." +
            commit_hash[:short_commit_hash_len] +
            ".epub"
        )

    if os.path.exists(out_epub):
        raise ValueError(f"output file exists: {out_epub!r}")

    # List tracked files in HEAD
    # Format: <mode> <type> <object> <path>
    entries = git(["ls-tree", "-r", "--full-tree", "HEAD"], text=True).splitlines()

    files = []
    for line in entries:
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        mode, typ, obj, path = parts
        if typ != "blob":
            continue
        files.append((mode, obj, path))

    paths = [p for _, _, p in files]

    if "mimetype" not in paths:
        fatal('"mimetype" file not tracked at HEAD (required for EPUB)')

    # Read all blobs
    blob_data = {}
    for _, obj, path in files:
        blob_data[path] = git(["cat-file", "blob", obj])

    # Read raw commit object
    commit_data = git(["cat-file", "commit", commit_hash])

    # Create EPUB (zip)
    with zipfile.ZipFile(out_epub, "w") as zf:
        # 1. mimetype (must be first, uncompressed)
        zi = zipinfo("mimetype", compress=zipfile.ZIP_STORED)
        zf.writestr(zi, blob_data["mimetype"])

        # 2. tracked files
        for path in sorted(blob_data.keys()):
            if path == "mimetype":
                continue
            if os.access(path, os.X_OK):
                mode = 0o755
            else:
                mode = 0o644
            zi = zipinfo(path, mode)
            zf.writestr(zi, blob_data[path])

        tree_hash = git(["rev-parse", "HEAD^{tree}"], text=True).strip()
        tree_bytes = git(
            ["ls-tree", "-r", "-t", "--full-tree", "HEAD"]
        )
        zi = zipinfo(f".git-tree-{tree_hash}.txt")
        zf.writestr(zi, tree_bytes)

        # 3. commit object
        zi = zipinfo(f".git-commit-{commit_hash}.txt")
        zf.writestr(zi, commit_data)

        # 4. git init script
        zi = zipinfo(".git-init.py", mode=0o755)
        zf.writestr(zi, generate_git_init_source().encode("utf-8"))

    print(out_epub)


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    if len(sys.argv) == 1:
        out_epub = None
    elif len(sys.argv) == 2:
        out_epub = sys.argv[1]
    else:
        print("usage: git2epub.py [output.epub]", file=sys.stderr)
        sys.exit(1)

    main(out_epub)
