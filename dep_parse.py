from multiprocessing import Pool
import threading
import os
import subprocess
import sys
import time, random
from multiprocessing import Lock

PLATFORM = 'iceberg'

if PLATFORM == 1:
    FILELIST = 'e:\\tmp\\filelist.txt'
    STANFORD_CORENLP_PATH = 'C:\\stanford-corenlp-full-2015-12-09'
    JAVA_HEAP_MEMORY = '2g'
elif PLATFORM == 2:
    FILELIST = '../tmp/filelist.txt'
    STANFORD_CORENLP_PATH = 'C:\\Users\\cop15rj\\PycharmProjects\\lexsub\\stanford-corenlp-full-2015-12-09'
    JAVA_HEAP_MEMORY = '2g'
    COMMANDS_FILE = '../tmp/commands.txt'
elif PLATFORM == 'iceberg':
    FILELIST = '/data/cop15rj/ukwac100/filelist.txt'
    STANFORD_CORENLP_PATH = '/home/cop15rj/lexsub/stanford-corenlp-full-2015-12-09'
    JAVA_HEAP_MEMORY = '2g'
    COMMANDS_FILE = '/data/cop15rj/ukwac100/commands.txt'

if sys.platform.find('win') != -1:
    JAVA_CP_SEP = ';'
else:
    JAVA_CP_SEP = ':'

QSUB_MEM = '8G'
QSUB_RMEM = '3G'
QSUB_MAIL_OPTION = 'n'
QSUB_MAIL_ADDRESS = 'rjain2@sheffield.ac.uk'
QSUB_REDIRECT_SCRIPT = 'redirect.bash'

PARSER_CMD = 'java -Xmx{0} ' \
             '-cp "{1}/*' + JAVA_CP_SEP + \
             '." ' \
             'edu.stanford.nlp.pipeline.StanfordCoreNLP ' \
             '-annotators tokenize,ssplit,pos,depparse ' \
             '-file {2} ' \
             '-outputDirectory {3} ' \
             '-outputFormat conll'

QSUB_REDIRECT_SCRIPT = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), QSUB_REDIRECT_SCRIPT)
QSUB_CMD = 'qsub -l mem=' + QSUB_MEM \
           + ' -l rmem=' + QSUB_RMEM \
           + ' -m ' + QSUB_MAIL_OPTION + ' -M ' + QSUB_MAIL_ADDRESS \
           + ' -j y -o {0} -N {1} -terse ' \
           + QSUB_REDIRECT_SCRIPT + ' {2}'

FILELIST = os.path.abspath(FILELIST)

commandsFile = open(os.path.abspath(COMMANDS_FILE), 'w')
threadLock = Lock()


def run_parser(inputFile):
    print('parsing input file', ':', inputFile)

    outputPath = os.path.dirname(inputFile)
    parserCmd = PARSER_CMD.format(JAVA_HEAP_MEMORY, STANFORD_CORENLP_PATH, inputFile, outputPath)

    print('parserCmd', ':', parserCmd)

    if PLATFORM == 'iceberg':
        qsubLogFile = os.path.join(os.path.dirname(inputFile), 'log_' + os.path.basename(inputFile))
        # qsubLogFile = os.path.join(os.path.dirname(inputFile), 'qsub_log')
        jobName = os.path.basename(inputFile)
        parserCmd = QSUB_CMD.format(qsubLogFile, jobName, parserCmd)

        print('qsubCmd', ':', parserCmd)

    threadLock.acquire()
    commandsFile.write(parserCmd)
    commandsFile.write('\n')
    commandsFile.flush()
    threadLock.release()

    parserProcess = subprocess.Popen(parserCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for line in parserProcess.stdout:
        print(os.path.basename(inputFile), ':', line.decode('utf-8'))

    parserProcess.stdout.close()
    parserProcess.wait()

    return


def depparse_input():
    fileList = open(FILELIST)

    inputFileNames = [line.strip() for line in fileList.readlines()]
    print('inputFileNames', ':', inputFileNames)
    for file in inputFileNames:
        print(file)

    pool = Pool()
    pool.map(run_parser, inputFileNames)

    pool.close()
    pool.join()

    fileList.close()
    commandsFile.close()


if __name__ == '__main__':
    depparse_input()
