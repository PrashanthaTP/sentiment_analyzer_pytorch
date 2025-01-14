negations_dic = {"isn't": "is not",     "aren't": "are not",    "wasn't": "was not",
                 "weren't": "were not", "haven't": "have not",  "hasn't": "has not",
                 "hadn't": "had not",     "won't": "will not", "wouldn't": "would not",
                 "don't": "do not",     "doesn't": "does not",   "didn't": "did not",
                 "can't": "can not",    "couldn't": "could not",  "shouldn't": "should not",
                 "mightn't": "might not",  "mustn't": "must not"}


CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",


}

SLANG_WORDS = {
    "AFAIK" : "As Far As I Know",
    "AFK" : "Away From Keyboard",
    "ASAP" : "As Soon As Possible",
    "ATK" : "At The Keyboard",
    "ATM" :" At The Moment",
    "A3" : "Anytime, Anywhere, Anyplace",
    "BAK" : "Back At Keyboard",
    "BBL" : "Be Back Later",
    "BBS" : "Be Back Soon",
    "BFN" : "Bye For Now",
    "B4N" : "Bye For Now",
    "BRB" : "Be Right Back",
    "BRT" : "Be Right There",
    "BTW" : "By The Way",
    "B4" : "Before",
    "B4N" : "Bye For Now",
    "CU" : "See You",
    "CUL8R" : "See You Later",
    "CYA" : "See You",
    "FAQ" : "Frequently Asked Questions",
    "FB" : "Facebook",
    "FC" : "Fingers Crossed",
    "FWIW" :" For What It's Worth",
    "FYI" : "For Your Information",
    "GAL" : "Get A Life",
    "GG" : "Good Game",
    "GN" : "Good Night",
    "GMTA" : "Great Minds Think Alike",
    "GR8" : "Great!",
    "G9" : "Genius",
    "IC" : "I See",
    "ICQ" :"I Seek you",
    "ILU": "I Love You",
    "IMHO" : "In My Honest/Humble Opinion",
    "IMO" : "In My Opinion",
    "IOW": "In Other Words",
    "IRL" : "In Real Life",
    "KISS" : "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO" : "Laugh My A.. Off",
    "LOL" : "Laughing Out Loud",
    "LTNS" :" Long Time No See",
    "L8R" : "Later",
    "MTE" : "My Thoughts Exactly",
    "M8" :" Mate",
    "NRN" :" No Reply Necessary",
    "OIC" : "Oh I See",
    "PITA" : "Pain In The A..",
    "PRT" : "Party",
    "PRW" :" Parents Are Watching",
    "QPSA":	"Que Pasa",
    "ROFL" : "Rolling On The Floor Laughing",
    "ROFLOL" : "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO" : "Rolling On The Floor Laughing My A.. Off",
    "SK8" : "Skate",
    "STATS": "Your sex and age",
    "ASL" : "Age, Sex, Location",
    "THX" : "Thank You",
    "TTFN" : "Ta-Ta For Now!",
    "TTYL" : "Talk To You Later",
    "U" : "You",
    "U2": "You Too",
    "U4E" : "Yours For Ever",
   "WB": "Welcome Back",
    "WTF" : "What The F...",
    "WTG" : "Way To Go!",
    "WUF" : "Where Are You From?",
    "W8" :" Wait",
   
}
# -*- coding: UTF-8 -*-


"""
Emoticons and Emoji data dictonary
"""


__all__ = ['EMO_UNICODE', 'UNICODE_EMO', 'EMOTICONS', 'EMOTICONS_EMO']

EMOTICONS = {
    u":‑\)": "Happy face or smiley",
    u":\)": "Happy face or smiley",
    u":-\]": "Happy face or smiley",
    u":\]": "Happy face or smiley",
    u":-3": "Happy face smiley",
    u":3": "Happy face smiley",
    u":->": "Happy face smiley",
    u":>": "Happy face smiley",
    u":\)": "Happy face smiley",
    u"8-\)": "Happy face smiley",
    u":o\)": "Happy face smiley",
    u":-\}": "Happy face smiley",
    u":\}": "Happy face smiley",
    u":-\)": "Happy face smiley",
    u":c\)": "Happy face smiley",
    u":\^\)": "Happy face smiley",
    u":\]": "Happy face smiley",
    u":\)": "Happy face smiley",
    u":‑D": "Laughing, big grin or laugh with glasses",
    u":D": "Laughing, big grin or laugh with glasses",
    u"8‑D": "Laughing, big grin or laugh with glasses",
    u"8D": "Laughing, big grin or laugh with glasses",
    u"X‑D": "Laughing, big grin or laugh with glasses",
    u"XD": "Laughing, big grin or laugh with glasses",
    u":D": "Laughing, big grin or laugh with glasses",
    u":3": "Laughing, big grin or laugh with glasses",
    u"B\^D": "Laughing, big grin or laugh with glasses",
    u":-\)\)": "Very happy",
    u":‑\(": "Frown, sad, andry or pouting",
    u":-\(": "Frown, sad, andry or pouting",
    u":\(": "Frown, sad, andry or pouting",
    u":‑c": "Frown, sad, andry or pouting",
    u":c": "Frown, sad, andry or pouting",
    u":‑<": "Frown, sad, andry or pouting",
    u":<": "Frown, sad, andry or pouting",
    u":‑\[": "Frown, sad, andry or pouting",
    u":\[": "Frown, sad, andry or pouting",
    u":-\|\|": "Frown, sad, andry or pouting",
    u">:\[": "Frown, sad, andry or pouting",
    u":\{": "Frown, sad, andry or pouting",
    u":@": "Frown, sad, andry or pouting",
    u">:\(": "Frown, sad, andry or pouting",
    u":'‑\(": "Crying",
    u":'\(": "Crying",
    u":'‑\)": "Tears of happiness",
    u":'\)": "Tears of happiness",
    u"D‑':": "Horror",
    u"D:<": "Disgust",
    u"D:": "Sadness",
    u"D8": "Great dismay",
    u"D;": "Great dismay",
    u"D:": "Great dismay",
    u"DX": "Great dismay",
    u":‑O": "Surprise",
    u":O": "Surprise",
    u":‑o": "Surprise",
    u":o": "Surprise",
    u":-0": "Shock",
    u"8‑0": "Yawn",
    u">:O": "Yawn",
    u":-\*": "Kiss",
    u":\*": "Kiss",
    u":X": "Kiss",
    u";‑\)": "Wink or smirk",
    u";\)": "Wink or smirk",
    u"\*-\)": "Wink or smirk",
    u"\*\)": "Wink or smirk",
    u";‑\]": "Wink or smirk",
    u";\]": "Wink or smirk",
    u";\^\)": "Wink or smirk",
    u":‑,": "Wink or smirk",
    u";D": "Wink or smirk",
    u":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑\|": "Straight face",
    u":\|": "Straight face",
    u":$": "Embarrassed or blushing",
    u":‑x": "Sealed lips or wearing braces or tongue-tied",
    u":x": "Sealed lips or wearing braces or tongue-tied",
    u":‑#": "Sealed lips or wearing braces or tongue-tied",
    u":#": "Sealed lips or wearing braces or tongue-tied",
    u":‑&": "Sealed lips or wearing braces or tongue-tied",
    u":&": "Sealed lips or wearing braces or tongue-tied",
    u"O:‑\)": "Angel, saint or innocent",
    u"O:\)": "Angel, saint or innocent",
    u"0:‑3": "Angel, saint or innocent",
    u"0:3": "Angel, saint or innocent",
    u"0:‑\)": "Angel, saint or innocent",
    u"0:\)": "Angel, saint or innocent",
    u":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;\^\)": "Angel, saint or innocent",
    u">:‑\)": "Evil or devilish",
    u">:\)": "Evil or devilish",
    u"\}:‑\)": "Evil or devilish",
    u"\}:\)": "Evil or devilish",
    u"3:‑\)": "Evil or devilish",
    u"3:\)": "Evil or devilish",
    u">;\)": "Evil or devilish",
    u"\|;‑\)": "Cool",
    u"\|‑O": "Bored",
    u":‑J": "Tongue-in-cheek",
    u"#‑\)": "Party all night",
    u"%‑\)": "Drunk or confused",
    u"%\)": "Drunk or confused",
    u":-###..": "Being sick",
    u":###..": "Being sick",
    u"<:‑\|": "Dump",
    u"\(>_<\)": "Troubled",
    u"\(>_<\)>": "Troubled",
    u"\(';'\)": "Baby",
    u"\(\^\^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(\^_\^;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(~_~;\) \(・\.・;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"\(-_-\)zzz": "Sleeping",
    u"\(\^_-\)": "Wink",
    u"\(\(\+_\+\)\)": "Confused",
    u"\(\+o\+\)": "Confused",
    u"\(o\|o\)": "Ultraman",
    u"\^_\^": "Joyful",
    u"\(\^_\^\)/": "Joyful",
    u"\(\^O\^\)／": "Joyful",
    u"\(\^o\^\)／": "Joyful",
    u"\(__\)": "Kowtow as a sign of respect, or dogeza for apology",
    u"_\(\._\.\)_": "Kowtow as a sign of respect, or dogeza for apology",
    u"<\(_ _\)>": "Kowtow as a sign of respect, or dogeza for apology",
    u"<m\(__\)m>": "Kowtow as a sign of respect, or dogeza for apology",
    u"m\(__\)m": "Kowtow as a sign of respect, or dogeza for apology",
    u"m\(_ _\)m": "Kowtow as a sign of respect, or dogeza for apology",
    u"\('_'\)": "Sad or Crying",
    u"\(/_;\)": "Sad or Crying",
    u"\(T_T\) \(;_;\)": "Sad or Crying",
    u"\(;_;": "Sad of Crying",
    u"\(;_:\)": "Sad or Crying",
    u"\(;O;\)": "Sad or Crying",
    u"\(:_;\)": "Sad or Crying",
    u"\(ToT\)": "Sad or Crying",
    u";_;": "Sad or Crying",
    u";-;": "Sad or Crying",
    u";n;": "Sad or Crying",
    u";;": "Sad or Crying",
    u"Q\.Q": "Sad or Crying",
    u"T\.T": "Sad or Crying",
    u"QQ": "Sad or Crying",
    u"Q_Q": "Sad or Crying",
    u"\(-\.-\)": "Shame",
    u"\(-_-\)": "Shame",
    u"\(一一\)": "Shame",
    u"\(；一_一\)": "Shame",
    u"\(:_:\)": "Tired",
    u"\(:\^\·\^:\)": "cat",
    u"\(:\^\·\·\^:\)": "cat",
    u":_\^:	": "cat",
    u"\(\.\.\)": "Looking down",
    u"\(\._\.\)": "Looking down",
    u"\^m\^": "Giggling with hand covering mouth",
    u"\(\・\・?": "Confusion",
    u"\(?_?\)": "Confusion",
    u">\^_\^<": "Normal Laugh",
    u"<\^!\^>": "Normal Laugh",
    u"\^/\^": "Normal Laugh",
    u"\（\*\^_\^\*）": "Normal Laugh",
    u"\(\^<\^\) \(\^\.\^\)": "Normal Laugh",
    u"\(^\^\)": "Normal Laugh",
    u"\(\^\.\^\)": "Normal Laugh",
    u"\(\^_\^\.\)": "Normal Laugh",
    u"\(\^_\^\)": "Normal Laugh",
    u"\(\^\^\)": "Normal Laugh",
    u"\(\^J\^\)": "Normal Laugh",
    u"\(\*\^\.\^\*\)": "Normal Laugh",
    u"\(\^—\^\）": "Normal Laugh",
    u"\(#\^\.\^#\)": "Normal Laugh",
    u"\（\^—\^\）": "Waving",
    u"\(;_;\)/~~~": "Waving",
    u"\(\^\.\^\)/~~~": "Waving",
    u"\(-_-\)/~~~ \($\·\·\)/~~~": "Waving",
    u"\(T_T\)/~~~": "Waving",
    u"\(ToT\)/~~~": "Waving",
    u"\(\*\^0\^\*\)": "Excited",
    u"\(\*_\*\)": "Amazed",
    u"\(\*_\*;": "Amazed",
    u"\(\+_\+\) \(@_@\)": "Amazed",
    u"\(\*\^\^\)v": "Laughing,Cheerful",
    u"\(\^_\^\)v": "Laughing,Cheerful",
    u"\(\(d[-_-]b\)\)": "Headphones,Listening to music",
    u'\(-"-\)': "Worried",
    u"\(ーー;\)": "Worried",
    u"\(\^0_0\^\)": "Eyeglasses",
    u"\(\＾ｖ\＾\)": "Happy",
    u"\(\＾ｕ\＾\)": "Happy",
    u"\(\^\)o\(\^\)": "Happy",
    u"\(\^O\^\)": "Happy",
    u"\(\^o\^\)": "Happy",
    u"\)\^o\^\(": "Happy",
    u":O o_O": "Surprised",
    u"o_0": "Surprised",
    u"o\.O": "Surpised",
    u"\(o\.o\)": "Surprised",
    u"oO": "Surprised",
    u"\(\*￣m￣\)": "Dissatisfied",
    u"\(‘A`\)": "Snubbed or Deflated"
}
