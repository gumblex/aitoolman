import json
import base64
import socket
import mimetypes
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import ksuid


def get_id():
    return str(ksuid.Ksuid())


def get_host_id(hostname=None):
    host_part = hostname or socket.gethostname()
    return host_part + "_" + str(ksuid.Ksuid())[-6:]


def make_insert(d):
    keys, values = zip(*d.items())
    return ', '.join(keys), ', '.join('?' * len(values)), values


def make_insert_auto(d):
    keys, values = zip(*d.items())
    converted_values = []
    for value in values:
        if (value is None or isinstance(value, str)
                or isinstance(value, int) or isinstance(value, float)):
            converted_values.append(value)
        else:
            converted_values.append(json.dumps(value, ensure_ascii=False))
    return ', '.join(keys), ', '.join('?' * len(values)), converted_values


def load_config(filename):
    with open(filename, 'rb') as f:
        return tomllib.load(f)


def load_config_str(s: str):
    return tomllib.loads(s)


def generate_data_url(data, content_type):
    return 'data:%s;base64,%s' % (
        content_type,
        base64.b64encode(data).decode('utf-8')
    )


def get_mime_type(filename):
    return mimetypes.guess_type(filename)[0]


def encode_message(obj):
    return json.dumps(obj, ensure_ascii=False).encode('utf-8')

