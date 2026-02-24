# Source - https://stackoverflow.com/a/34325723
# Posted by Greenstick, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-23, License - CC BY-SA 4.0
# Heavily modified by me

class ProgressBar:
    def __init__(self, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '⬤', print_end = "\r"):
        """
        Create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        self.i = 0
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end

        self.print_bar()

    def print_bar(self, i=None):
        if i == None: i = self.i
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.i / float(self.total)))
        filled_len = int(self.length * self.i // self.total)
        bar = self.fill * filled_len + '⋅' * (self.length - filled_len)
        print(f'\r{self.prefix} ({bar}) [{self.i}/{self.total}] {percent}% {self.suffix}', end = self.print_end)
        if self.i == self.total: print() # newline if complete

    def iterate(self, i=None):
        if i == None: i = self.i+1
        self.i = i
        self.print_bar()
        