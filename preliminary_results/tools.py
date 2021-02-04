import Dict, List, Set

Count = int
Class, Document, Word = str, str, str

def extract(
    data : Dict[ Class, Set[ Document ] ],
    pattern : str = '\\b[a-z]{2,}\\b'
    ) -> Dict[ Word, Count ] :
    '''
    Extract Vocabulary from Dataset
    '''
    vocabulary : Dict[ Word, Count ] = {}
    for label in data :
        for document in data[ label ] :
            for word in re.findall(
                pattern = pattern,
                string  = document
                ) :
                if word not in self.stop_words :
                    if word in vocabulary :
                        vocabulary[ word ] += 1
                    else :
                        vocabulary[ word ] = 1
    return vocabulary

def calcStats(
    predictions : Dict[ Class, Dict[ Class, Count ] ],
    accuracy : List[ float ],
    ntrained : int,
    ntested  : int
    ) -> None :
    # Calculating and displaying stats
    # True Positive Rate : True Positives / Actual Positives
    # False Positive Rate: False Positive / Actual Negatives
    # Precision: True Positives / ( True Positives + False Positives )
    self.predictions = predictions.copy()
    for label in predictions :
        # predictions :
        # Dict[ Actual Class, Dict[ Output Class, Count ] ]
        actual = 0
        total_correct = 0
        truth_plus_false = 0
        for label_ in predictions :
            actual += predictions[ label ][ label_ ]
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
    if len( accuracy ) > 1 :
        mean = sum( accuracy ) / len( accuracy )
        stdv = sum( ( x - mean )**2 / ( len( accuracy ) - 1 )
                     for x in accuracy )**0.5
        deci = self.__decimalPlace( stdv )
    else :
        mean, stdv, deci = self.accuracy, 0, 2
    stdv = round( stdv, deci )
    accuracy = round( mean, deci )
    self.predictions[ 'model' ] = {}
    self.predictions[ 'model' ][ 'accuracy' ] = '{}({})'.format( accuracy, stdv )
    print( 'Examples Trained:', ntrained )
    print( 'Examples Tested :', ntested )
    print( 'Total Examples  :', ntrained + ntested )
    return

def removeFirstQ( vocabulary : Dict[ Word, Count ] ) -> Dict[ Word, Count ] :
    '''
    Remove First Quartile in a Vocabulary
    '''
    if not vocabulary : return vocabulary
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
    return vocabulary

def decimalPlace( self, n : Union[ int, str ] ) -> int :
    '''
    Find Decimal Place of First Significant Figure
    '''
    if not n :
        return 0
    n = str( n )
    if '.' in n :
        i, f = str( n ).split( '.' )
        if i != '0' :
            return -len( i )
        rslt = 0
        for digit in f :
            rslt += 1
            if digit != '0' :
                return rslt
    return -len( n )
