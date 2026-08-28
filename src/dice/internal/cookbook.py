import re
from pathlib import Path
from urllib.parse import urlparse

import ujson
from sqlmodel import or_, select

from dice.shared.query import query, to_sql

from .models import RecipeRef
from .recipe import Descriptor
from .repository import Repository

IDENTIFIER_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def is_local(value: str) -> bool:
    return Path(value).exists()

def is_remote(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False

    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def is_identifier(value: str) -> bool:
    return bool(IDENTIFIER_RE.fullmatch(value))

class Cookbook:

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.desc_fname: str = "recipe.json"

    def ref(self, identifier: str) -> tuple[RecipeRef | None, Exception | None]:
        q = query(RecipeRef, clauses={"name":identifier})
        entry = self.repo.search(q).one()
        if not entry:
            return None, None

        path = Path(entry["path"])
        if not path.is_dir():
            return None, FileNotFoundError(f"Recipe directory not found: {identifier}")

        if not (path / self.desc_fname).is_file():
            return None, FileNotFoundError(f"Recipe descriptor not found: {identifier}")

        return RecipeRef(
            name=identifier,
            path=str(path),
        ), None

    def find(self, recipes: list[str] | None = None) -> list[RecipeRef]:
        patterns = recipes or ["*"]

        conditions = [
            RecipeRef.name.op("GLOB")(pattern)
            for pattern in patterns
        ]
        statement = select(RecipeRef).where(or_(*conditions))

        ret = []
        for res in self.repo.search(to_sql(statement)):
            ref, _ = self.ref(res.id)
            if ref is not None:
                ret.append(ref)

        return ret

    def read_recipe(self, fpath: str | Path) -> Descriptor:
        if isinstance(fpath, str):
            fpath = Path(fpath)
        if fpath.is_dir():
            fpath = fpath / self.desc_fname
        if not fpath.exists():
            raise FileNotFoundError(f"recipe not found: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = ujson.load(f)
            return Descriptor(**data)

    def fetch_recipe(self, id: str) -> Descriptor:
        # This function should pull the recipe from the repo. It does not save it locally.
        raise NotImplementedError

    def pull_recipe(self, url: str) -> Descriptor:
        raise NotImplementedError

    def resolve(self, fpath: str) -> Descriptor:
        if is_local(fpath):
            return self.read_recipe(fpath)

        if is_identifier(fpath):
            ref, _ = self.ref(fpath)
            if ref is not None:
                assert ref.path is not None
                return self.read_recipe(Path(ref.path) / self.desc_fname)

            return self.fetch_recipe(fpath)

        if is_remote(fpath):
            return self.pull_recipe(fpath)

        raise ValueError(f"unable to resolve recipe location: {fpath}")


    def materialize(
        self,
        ref: RecipeRef,
        destination: str | Path,
    ) -> RecipeRef:
        raise NotImplementedError

def new_cookbook(repo: Repository) -> Cookbook:
    return Cookbook(repo)
