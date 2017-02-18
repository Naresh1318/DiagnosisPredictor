"""
Functions for converting among ICD9 formats.
"""

def _zero_pad(x, n=3):
    if len(x) < n:
        x = (n - len(x)) * "0" + x
    return x


def decimal_to_parts(code):
    """
    Convert an ICD9 code from decimal format to major and minor parts.
    """

    parts = code.split(".")
    if len(parts) == 2:
        major, minor = parts[0], parts[1]
    else:
        major, minor = parts[0], ""

    if major[0] == "E":
        major = "E" + _zero_pad(major[1:])

    elif major[0] == "V":
        major = "V" + _zero_pad(major[1:3], 2)

    else:
        major = _zero_pad(major)

    return major, minor


def decimal_to_short(code):
    """
    Convert an ICD9 code from decimal format to short format.
    """

    parts = code.split(".")
    parts[0] = _zero_pad(parts[0])

    return "".join(parts)


def short_to_decimal(code):
    """
    Convert an ICD9 code from short format to decimal format.
    """

    if len(code) <= 3:
        return code.lstrip("0")
    else:
        return code[:3].lstrip("0") + "." + code[3:]


def short_to_parts(code):
    """
    Convert an ICD9 code from short format to the major and minor parts.
    """

    if code[0] == "E":
        major = "E" + _zero_pad(code[1:4])

        if len(code) > 4:
            minor = code[4:]
        else:
            minor = ""

    elif code[0] == "V":
        major = "V" + _zero_pad(code[1:3])

        if len(code) > 3:
            minor = code[3:]
        else:
            minor = ""

    else:
        major, minor = code[:3], code[3:]

    return major, minor


def parts_to_short(major, minor):
    """
    Convert an ICD9 code from major/minor parts to short format.
    """

    if major[0]  == "E":
        major = major[0] + _zero_pad(major[1:])
    elif major[0]  == "V":
        major = major[0] + _zero_pad(major[1:], 2)
    else:
        major = _zero_pad(major)

    return major + minor



def parts_to_decimal(major, minor):
    """
    Convert an ICD9 code from major/minor parts to decimal format.
    """

    if major[0] in ("V", "E"):
        major = major[0] + major[1:].lstrip("0")
        if len(major) == 1:
            major = major + "0"
    else:
        major = major.lstrip("0")
        if len(major) == 0:
            major = "0"

    return major + "." + minor


