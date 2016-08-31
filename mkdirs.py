import os
import errno


def mkdirs(path):
    if not os.path.exists(path):
        try:
            if os.path.isfile(path):
                print(path)
                os.makedirs(os.path.dirname(path))
            else:
                print(path)
                os.makedirs(path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise