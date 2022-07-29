import logging

formats = {'small': logging.Formatter('%(asctime)s - %(message)s', datefmt='%m-%d %H:%M'),
           'long': logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M')}

def create_log(logname, logdir, logformat='small', console=False):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)

    formatter = formats[logformat]

    fh = logging.FileHandler(logdir / 'train.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    return logger