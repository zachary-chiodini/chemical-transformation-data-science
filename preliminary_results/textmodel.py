import numpy as np
from typing import Dict, List, Tuple

class TextModel :
    '''
    Common Features of a Text Classification Model
    '''
    # data structures
    Count = int
    Array = List
    Class, Document, Word = str, str, str
    Data = Array[ Tuple[ Document, Class ] ]
    Vocabulary = Array[ Word ]

    def __init__( self ) -> None :
        self.label       : List[ Class ] = []
        self.accuracy    : List[ float ] = []
        self.ntrained    : int = 0
        self.ntested     : int = 0
        self.vocabulary  : Vocabulay = np.array([])
        self.predictions : Dict[ Class, Dict[ Class, Count ] ] = {}
        self.stop_words = {
            'if', 'might', 'big', 'opens', 'but', 'got',
            'almost', 'differently', 'since', 'why', 'things',
            'under', 'perhaps', 'grouped', 'whose', 'show',
            'say', 'first', 'us', 'used', 'room', 'that',
            'seems', 'groups', 'over', 'we', 'whether',
            'wants', 'thus', 'number', 'four', 'you',
            'anywhere', 'smaller', 'within', 'man', 'already',
            'may', 'second', 'though', 'to', 'furthering',
            'finds', 'across', 'through', 'give', 'needing',
            'turns', 'see', 'really', 'interesting', 'what',
            'while', 'ever', 'yet', 'latest', 'greater', 'be',
            'forth', 'nowhere', 'end', 'is', 'on', 'everyone',
            'clear', 'where', 'all', 'area', 'anybody', 'turn',
            'made', 'your', 'ten', 'place', 'faces', 'our',
            'here', 'clearly', 'than', 'something', 'downed',
            'sure', 'asks', 'backs', 'seven', 'everywhere',
            'parts', 'went', 'herself', 'youngest', 'important',
            'put', 'uses', 'around', 'until', 'during', 'these',
            'younger', 'just', 'whole', 'come', 'became',
            'large', 'off', 'mrs', 'seventh', 'must', 'being',
            'presenting', 'rooms', 'his', 'toward', 'into',
            'early', 'each', 'possible', 'later', 'everybody',
            'open', 'highest', 'was', 'more', 'interested',
            'backed', 'several', 'showing', 'most', 'interests',
            'year', 'certain', 'well', 'fully', 'have',
            'wanting', 'last', 'double', 'against', 'down',
            'high', 'had', 'still', 'far', 'point', 'ordering',
            'began', 'good', 'puts', 'then', 'says', 'opening',
            'any', 'evenly', 'given', 'long', 'who', 'because',
            'again', 'away', 'others', 'downs', 'and', 'largely',
            'three', 'nine', 'worked', 'higher', 'let', 'both',
            'right', 'shows', 'has', 'taken', 'become', 'sees',
            'men', 'my', 'making', 'problem', 'back', 'been',
            'how', 'did', 'among', 'it', 'mostly', 'per',
            'places', 'turning', 'saw', 'sides', 'himself',
            'out', 'part', 'cases', 'opened', 'ending',
            'although', 'seeming', 'member', 'therefore',
            'pointed', 'fact', 'older', 'took', 'ended', 'two',
            'other', 'above', 'once', 'keeps', 'think', 'eighth',
            'there', 'use', 'enough', 'does', 'few', 'every',
            'longest', 'somebody', 'way', 'present', 'needed',
            'new', 'do', 'felt', 'side', 'he', 'interest', 'make',
            'such', 'even', 'nothing', 'knows', 'needs', 'goods',
            'itself', 'final', 'working', 'best', 'much',
            'points', 'when', 'of', 'she', 'no', 'respectively',
            'which', 'five', 'yours', 'thing', 'should', 'many',
            'full', 'next', 'twice', 'oldest', 'longer', 'for',
            'without', 'upon', 'seem', 'anything', 'backing',
            'in', 'they', 'lets', 'so', 'smallest', 'came',
            'cannot', 'said', 'someone', 'general', 'showed',
            'from', 'less', 'downing', 'noone', 'tenth', 'sixth',
            'case', 'wanted', 'works', 'thoughts', 'numbers',
            'need', 'different', 'find', 'shall', 'knew', 'parting',
            'wells', 'facts', 'together', 'him', 'myself', 'gets',
            'ways', 'least', 'having', 'one', 'after', 'areas',
            'take', 'ends', 'get', 'this', 'either', 'members',
            'small', 'kind', 'six', 'great', 'some', 'would',
            'quite', 'however', 'like', 'years', 'state', 'or',
            'face', 'also', 'furthers', 'non', 'behind', 'group',
            'know', 'with', 'them', 'necessary', 'order', 'rather',
            'will', 'along', 'by', 'generally', 'gave', 'thought',
            'presents', 'further', 'grouping', 'greatest', 'old',
            'between', 'nobody', 'third', 'mr', 'those', 'eight',
            'thinks', 'work', 'alone', 'furthered', 'their', 'the',
            'seemed', 'her', 'newer', 'ninth', 'can', 'about',
            'before', 'only', 'go', 'likely', 'not', 'gives',
            'presented', 'another', 'at', 'parted', 'newest', 'very',
            'ask', 'asking', 'turned', 'states', 'fifth', 'beings',
            'up', 'often', 'same', 'known', 'problems', 'differ',
            'somewhere', 'keep', 'certainly', 'anyone', 'too',
            'going', 'want', 'me', 'seconds', 'never', 'triple',
            'young', 'its', 'as', 'everything', 'were', 'asked',
            'done', 'pointing', 'ordered', 'an', 'orders', 'could',
            'better', 'are', 'becomes', 'today', 'always', 'now'
            }
        return

    def test( self, data : Data ) -> None :
        '''
        Test Classification Model on Data
        '''
        accuracy = 0.0
        self.ntested = 0
        for label in data :
            self.predictions[ label ] = {
                label : 0 for label in data
                }
            for document in data[ label ] :
                self.ntested += 1
                output = self.output( document )
                accuracy += ( label == output )
                self.predictions[ label ][ output ] += 1
        if tot_docs :
            self.accuracy = [ accuracy / self.ntested ]
        else :
            self.accuracy = [ 0 ]
        return

    def getStats( self ) -> None :
        '''
        Calculate and Display Model Stats
        ---------------------------------
        (1) True Positive (TP) Rate:
                TP / Actual Positives
        (2) False Positive (FP) Rate:
                FP / Actual Negatives
        (3) Precision:
                TP / ( TP + FP )
        '''
        for label in self.predictions :
            actual, total_correct, truth_plus_false = 0, 0, 0
            for label_ in self.predictions :
                actual += self.predictions[ label ][ label_ ]
                total_correct += predictions[ label_ ][ label_ ]
                truth_plus_false += predictions[ label_ ][ label ]
            truth = predictions[ label ][ label ]
            false = truth_plus_false - truth
            self.predictions[ label ][ 'truth rate' ] = (
                truth / actual
                )
            self.predictions[ label ][ 'false rate' ] = (
                false / ( total_correct - truth + false )
                )
            self.predictions[ label ][ 'precision'  ] = (
                truth / truth_plus_false
                )
        mean = sum( self.accuracy ) / len( self.accuracy )
        stdv = sum( ( x - mean )**2 / ( len( self.accuracy ) - 1 )
                    for x in self.accuracy )**0.5
               if len( self.accuracy ) > 1
               else 0
        deci = self.__decimal( stdv )
        stdv = round( stdv, deci )
        self.accuracy = round( mean, deci )
        self.predictions[ 'model' ] = {}
        self.predictions[ 'model' ][ 'accuracy' ] = (
            '{}({})'.format( self.accuracy, stdv )
            )
        print( 'Examples Trained:', self.ntrained )
        print( 'Examples Tested :', self.ntested )
        print( 'Total Examples  :', self.ntrained + self.ntested )
        return

    def __decimal( N : Union[ int, str ] ) -> int :
        '''
        Returns the Decimal Place of the First SigFig in N
        '''
        match = re.search( '^[1-9][0-9]*|0\.0*[1-9]', str( N ) )
        if match :
            match = match.group()
            if '0.' in match :
                return len( match ) - 2
            return -len( match )
        return 0

    def extract( data : Data ) -> Vocabulary :
        '''
        Extract Vocabulary from Data
        '''
        vocabulary : Dict[ Word, Count ] = {}
        for document, label in data :
            for word in re.findall(
                pattern = '\\b[a-z]{2,}\\b',
                string  = document
                ) :
                if word not in self.stop_words :
                    if word in vocabulary :
                        vocabulary[ word ] += 1
                    else :
                        vocabulary[ word ] = 1
        # remove first quartile in vocabulary
        count = sorted( set( vocabulary.values() ) )
        index = len( count ) / 4
        if index % 1 == 0 :
            limit = count[ int( index ) ]
        else :
            index = int( index ) # truncate
            limit = ( count[ index ] + count[ index + 1 ] ) / 2
        for word, count in vocabulary.copy().items() :
            if count < limit :
                del vocabulary[ word ]
        return list( vocabulary )

    
