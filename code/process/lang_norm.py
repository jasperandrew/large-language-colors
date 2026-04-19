import re

class LangNormalizer():
    def __init__(self, convert_fn=None, ignore_regex=None, term_end=None, replacements=None):
        self.convert_fn = convert_fn
        self.ignore_regex = ignore_regex
        self.term_end = term_end
        self.replacements = replacements

    def convert(self, term):
        if self.convert_fn == None: return term
        return self.convert_fn(term)

    def ignore(self, term):
        if self.ignore_regex == None: return term
        return re.sub(self.ignore_regex, '', term)

    def strip_end(self, term):
        if self.term_end != None and term.endswith(self.term_end):
            return term[:(-1*len(self.term_end))]
        return term

    def replace(self, term):
        if self.replacements == None: return term
        for repl,variants in self.replacements.items():
            for regex in variants:
                if re.search(regex, term):
                    term = re.sub(regex, repl, term)
        return term
    
    def process(self, term):
        term = self.convert(term)
        term = self.ignore(term)
        term = self.strip_end(term)
        term = self.replace(term)
        term = self.replace(term) # again to catch potential new matches
        return term


import rules.en as en
import rules.ko as ko
import rules.zh as zh

LANG_NORMALIZERS = {
    "en": LangNormalizer(ignore_regex=en.IGNORE_REGEX, replacements=en.REPLACEMENTS),
    "ko": LangNormalizer(ignore_regex=ko.IGNORE_REGEX, term_end=ko.TERM_END, replacements=ko.REPLACEMENTS),
    "zh": LangNormalizer(convert_fn=zh.CONVERT_FN, ignore_regex=zh.IGNORE_REGEX, term_end=zh.TERM_END, replacements=zh.REPLACEMENTS),
}

NO_NORMALIZER = LangNormalizer()

def get_normalizer(lang_code):
    if lang_code in LANG_NORMALIZERS.keys(): return LANG_NORMALIZERS[lang_code]
    return NO_NORMALIZER