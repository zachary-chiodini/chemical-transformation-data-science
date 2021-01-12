import random, re
from typing import Dict, List, Optional, Set, Union

class LogisticRegression :
    '''
    Binomial Logistic Regression for Text Classification
    '''
    Count = int
    Class, Document, Word = str, str, str
    Data = Dict[ Class, Set[ Document ] ]
    
    def __init__( self ) :
        self.label = []
        self.network = np.array([])    # weights and bias
        self.vocabulary = np.array([]) # features
        self.accuracy = 0.0
        self.predictions : Dict[ Class, Dict[ Class, int ] ] = {}
        self.__vecfindall = np.vectorize( 
            lambda voc, doc : int( voc in doc ) 
            )
        self.stop_words = {}
        with open( 'stopset.txt' ) as file :
            self.stop_words = set( file.read().split( ',' ) )
            
    def train( self, 
        data : Dict[ Class, Set[ Document ] ],
        target : Union[ str, int ],
        rate : float = 1,
        batches : int = 10,
        convergence : float = 0.01
        ) -> None :
        '''
        Logistic Regression Model Training Algorithm
        '''
        # index 1 for target class; index 0 for other class
        self.label = [ label for label in data if label != target ]
        self.label.append( target )
        # get features
        self.vocabulary = np.array( list( 
            self.__removeFirstQ( self.__extract( data ) )
            ) )
        # append bias
        self.vocabulary = np.append( self.vocabulary, '' )
        # generate random weights
        self.network = np.random.uniform( 
            -0.5, 0.5, size = len( self.vocabulary )
            )
        # generate input vectors
        X = np.empty( shape = ( 0, len( self.vocabulary ) ) )
        for label in data :
            for document in data[ label ] :
                X = np.vstack( ( X, self.getInputFrom( document ) ) )
        # get target values
        Y = np.array( [ int( label == target ) for label in data 
                        for document in data[ label ] ] )
        totgrad = np.inf
        # gradient descent training algorithm
        while abs( totgrad ) > convergence :
            totgrad = 0
            for xbatch, ybatch in zip( 
                np.array_split( X, batches ), 
                np.array_split( Y, batches ) 
                ) :
                grad = np.multiply( 
                    np.reshape( 
                        self.output( xbatch ) - ybatch, 
                        newshape = ( len( xbatch ), 1 )
                        ),
                    xbatch
                    ).sum( axis = 0 ) / len( xbatch )
                self.network = self.network - rate*grad
                totgrad += grad.sum() / len( grad )
        return
    
    def test( self, 
        data : Dict[ Class, Set[ Document ] ], 
        boundary : float = 0.5 
        ) -> None :
        '''
        Logistic Regression Model Testing Algorithm
        '''
        assert self.network.size != 0, \
            'A regression model must be trained.'
        tot_docs = 0
        accuracy = 0.0
        for label in data :
            self.predictions[ label ] = { 
                label : 0 for label in data 
            }
            for document in data[ label ] :
                tot_docs += 1
                output = self.label[ int( 
                    self.output( 
                        self.getInputFrom( document )
                        ) > boundary 
                    ) ]
                self.predictions[ label ][ output ] += 1
                accuracy += ( label == output )
        if tot_docs :
            self.accuracy = accuracy / tot_docs
        else :
            self.accuracy = 0
        return
    
    def getInputFrom( self, document : str ) -> NDArray[ int ] :
        ''' Generate Input Vector '''
        return self.__findall( self.vocabulary, document )
            
    def output( self, X : NDArray[ int ] ) :
        ''' Logistic Model Output '''
        return self.sigmoid( np.dot( X, self.network ) )
            
    def sigmoid( self, x : float ) -> float :
        return 1 / ( 1 + e**(-x) )
    
    def __findall( self, 
        vocabulary : NDArray[ Word ], 
        document   : str
        ) -> NDArray[ int ] :
        return self.__vecfindall( vocabulary, document )

    def __extract( self,
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
    
    def __removeFirstQ( self,
        vocabulary : Dict[ Word, Count ]
        ) -> Dict[ Word, Count ] :
        '''
        Remove First Quartile in Vocabulary
        '''
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
    
    def kFoldValidate( self,
        data : Dict[ Class, Set[ Document ] ],
        target : Union[ str, int ],
        rate : float = 1,
        batches : int = 10,
        convergence : float = 0.01,
        k : int = 10
        ) -> None :
        '''
        K-Fold Cross-Validation
        '''
        self.predictions = {}
        self.accuracy = 0.0
        accuracy = []
        ntested, ntrained = 0, 0
        predictions = {
            label : { label : 0  for label in data }
            for label in data
            }
        shuffled : Data = {}
        klist : List[ Data ] = []
        # shuffling data
        for label in data :
            temp = list( data[ label ] )
            random.shuffle( temp )
            shuffled[ label ] = temp
        # creating k sets of data
        temp = {}
        for i in range( 1, k + 1 ) :
            for label in shuffled :
                strt = int( round( ( ( i - 1 ) / k ) * len( shuffled[ label ] ), 0 ) )
                fnsh = int( round( ( i / k ) * len( shuffled[ label ] ), 0 ) )
                temp[ label ] = shuffled[ label ][ strt : fnsh ]
            klist.append( temp )
        # k-fold cross validation
        for i in range( len( klist ) ) :
            test = klist[ i ]
            for j in range( len( klist ) ) :
                if i == j :
                    continue
                train = klist[ j ]
                self.train( train, target, rate, batches, convergence )
                self.test( test )
                ntrained += sum( len( train[ label ] )
                                 for label in train )
                accuracy.append( self.accuracy )
                for label in self.predictions :
                    for predicted in self.predictions[ label ] :
                        predictions[ label ][ predicted ] += \
                            self.predictions[ label ][ predicted ]
            ntested += sum( len( test[ label ] )
                            for label in test )
        self.__calcStats( predictions, accuracy, ntrained, ntested )
        return

    def trainAndTest( self,
        data : Dict[ Class, Set[ Document ] ],
        target : Union[ str, int ],
        rate : float = 1,
        batches : int = 10,
        convergence : float = 0.01,
        ratio : float = 0.0,
        iters : int = 10
        ) -> None :
        '''
        Monte Carlo Cross-Validation
        '''
        assert 0 <= ratio < 1, 'Ratio is between 0 and 1.'
        self.predictions = {}
        self.accuracy = 0.0
        test, train = {}, {}
        ntested, ntrained = 0, 0
        accuracy = []
        predictions = {
            label : { label : 0  for label in data }
            for label in data
            }
        for _ in range( iters ) :
            split = ratio if ratio else random.random()
            for label in data :
                index = int( split * len( data[ label ] ) )
                while not index :
                    split = random.random()
                    index = int( split * len( data[ label ] ) )
                shuffle = list( data[ label ] )
                random.shuffle( shuffle )
                train[ label ] = shuffle[ : index ]
                test [ label ] = shuffle[ index : ]
                ntrained += len( train[ label ] )
                ntested += len( test[ label ] )
            self.train( train, target, rate, batches, convergence )
            self.test( test )
            accuracy.append( self.accuracy )
            for label in self.predictions :
                for predicted in self.predictions[ label ] :
                    predictions[ label ][ predicted ] += \
                        self.predictions[ label ][ predicted ]
        self.__calcStats( predictions, accuracy, ntrained, ntested )
        return

    def __calcStats( self,
        predictions : Dict[ Class, Dict[ Class, Count ] ],
        accuracy : List[ float ],
        ntrained : int, ntested : int
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
    
    def __decimalPlace( self, n : Union[ int, str ] ) -> int :
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
