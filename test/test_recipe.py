import logging

import ujson

from dice.internal.recipe import new_builder, unmarshal
from dice.shared.modules import ModuleEnum

logger = logging.getLogger(__name__)

class TestRecipe:
    def test_custom_recipe(self):
        recipe = (
            new_builder()
            .components([ModuleEnum.CLASSIFIER.value], ["example"])
            .params(example={"eg": "example"})
            .bake()
        )
        d1 = recipe.desc.to_dict()
        txt = ujson.dumps(d1)
        desc = unmarshal(ujson.loads(txt))
        d2 = desc.to_dict()
        assert d1 == d2
