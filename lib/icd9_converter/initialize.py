import json
from pkg_resources import resource_string

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def load_json(x):
    contents = resource_string(__name__, "resources/" + x)
    return json.loads(contents.decode('utf-8'))


ahrqComorbidAll  = load_json("ahrqComorbidAll.json")
elixComorbid     = load_json("elixComorbid.json")
icd9Hierarchy    = load_json("icd9Hierarchy.json")
ahrqComorbid     = load_json("ahrqComorbid.json")
icd9Chapters     = load_json("icd9Chapters.json")
quanElixComorbid = load_json("quanElixComorbid.json")
