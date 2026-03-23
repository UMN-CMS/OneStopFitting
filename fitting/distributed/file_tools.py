import os
import tarfile


def tarDirectory(
    path,
    output,
    skip_words=(".git", ".github", ".pytest_cache", "tests", "docs"),
    skip=(lambda fn: os.path.splitext(fn)[1] == ".pyc",),
    mode="w",
):
    with tarfile.open(output, f"{mode}:gz") as z:
        for root, dirs, files in os.walk(path):
            for file in files:
                filename = os.path.join(root, file)
                if any(predicate(filename) for predicate in skip):
                    continue
                dirs = filename.split(os.sep)
                if any(word in dirs for word in skip_words):
                    continue

                archive_name = os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                )
                z.add(filename, archive_name)


def tarFiles(
    paths,
    output,
    skip_words=(".git", ".github", ".pytest_cache", "tests", "docs"),
    skip=(lambda fn: os.path.splitext(fn)[1] == ".pyc",),
    mode="w",
):
    with tarfile.open(output, f"{mode}:gz") as z:
        for file in paths:
            z.add(file)
