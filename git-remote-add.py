#!/usr/bin/env python3

import subprocess
from urllib.parse import urlparse

# pretty remote names
DOMAIN_MAP = {
    # https://github.com/milahu/darknet-git-hosting-services
    "gg6zxtreajiijztyy5g6bt5o6l3qu32nrg7eulyemlhxwwl6enk6ghad.onion": "righttoprivacy.onion",
    "git.dkforestseeaaq2dqz2uflmlsybvnq2irzn4ygyvu53oazyorednviid.onion": "darkforest.onion",
    "it7otdanqu7ktntxzm427cba6i53w6wlanlh23v5i3siqmos47pzhvyd.onion": "darktea.onion",
    "gdatura24gtdy23lxd7ht3xzx6mi7mdlkabpvuefhrjn4t5jduviw5ad.onion": "gdatura.onion",
}

REPOS_FILE = "repos.txt"

def remote_exists(name):
    """Check if a git remote already exists"""
    result = subprocess.run(["git", "remote"], capture_output=True, text=True)
    return name in result.stdout.splitlines()

def add_remote(name, url):
    """Add a git remote if it doesn't already exist"""
    if remote_exists(name):
        print(f"keeping remote {name!r}")
        return

    print(f"adding remote {name!r}")
    subprocess.run(["git", "remote", "add", name, url], check=True)

    if urlparse(url).netloc.endswith(".onion"):
        subprocess.run([
            "git", "config", "--add",
            f'remote.{name}.proxy', "socks5h://127.0.0.1:9050"
        ], check=True)

def main():
    os.chdir(os.path.dirname(__file__))
    with open(REPOS_FILE) as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            domain = urlparse(url).netloc
            remote_name = DOMAIN_MAP.get(domain, domain)
            add_remote(remote_name, url)

if __name__ == "__main__":
    main()
