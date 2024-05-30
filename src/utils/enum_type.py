# NB:  File names must not conflict with python core otherwise can cause issues
# https://stackoverflow.com/questions/43124775/why-python-3-6-1-throws-attributeerror-module-enum-has-no-attribute-intflag/45716067#comment73328722_43124775
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type("Enum", (), enums)
