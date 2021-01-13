import codecs, funks as f
from random import shuffle
import matplotlib.pyplot as plt
from typing import Any, List, Set, Tuple

class DecisionTree :

    def __init__(
        self,
        label : List[ str ] = list( '' ),
        data  : Set[ Tuple[ Any ] ] = set( tuple( '' ) )
        ) -> None :
        self.label = label
        self.n     = 0
        self.data  = data
        self.tree  = {} # nested hash map
        self.goal  = []
        self.rslt  = []
        self.testset  = set( tuple( '' ) )
        self.trainset = set( tuple( '' ) )
        return

    def importcsv( self, path : str ) -> None :
        self.data = set( tuple( '' ) )
        with codecs.open(
            path, 'r', 'utf-8-sig'
            ) as file :
            self.label = file.readline().strip().split( ',' )
            self.n = len( self.label ) - 1
            for line in file :
                self.data.add(
                    tuple( line.strip().split( ',' ) )
                    )
        return

    def learn( self, data ) -> None :
        assert self.label, \
               'DecisionTree needs a list of labels.'
        self.tree = {}
        self.__recurse( data, self.tree )
        return
                    
    def __recurse(
        self,
        S : Set[ Tuple[ Any ] ],
        tree : 'Nested Dict',
        ) -> 'Nested Dict' :
        index = f.bestAttribute( S, self.n )
        split = f.split( S, index )
        if len( split ) == 1 :
            tree[ f.majorityVote( S ) ] = None
            return
        tree[ self.label[ index ] ] = {}
        node = tree[ self.label[ index ] ]
        for value in split :
            node[ value ] = {}
            self.__recurse(
                f.subset( S, value, index ),
                tree = node[ value ]
                )
        return

    def test( self, data ) -> None :
        assert self.tree, \
               'DecisionTree needs to learn a data set.'
        self.rslt, self.goal = [], []
        for sample in data :
            self.rslt.append(
                self.output( sample, self.tree )
                )
            self.goal.append( sample[ -1 ] )
        return

    def output(
        self,
        sample : Tuple[ Any ],
        tree   : 'Nested Dict'
        ) -> Any :
        node = list( tree.keys() )[ 0 ]
        if node in self.label :
            index = self.label.index( node )
            value = sample[ index ]
            if value in tree[ node ].keys() :
                node = self.output( sample, tree[ node ][ value ] )
            else :
                return 'Unknown Value'
        return node

    def showResult( self ) -> None :
        tally = 0
        for i in range( len( self.rslt ) ) :
            if self.rslt[ i ] == self.goal[ i ] :
                tally += 1
        if self.rslt :
            tally = round( 100 * tally / len( self.rslt ), 2 )
        else :
            tally = 'Unknown'
        print( 'Model accuracy         : ', tally, '%' )
        return 

    def testAndTrain( self, ratio : float = 0.5 ) :
        assert self.data and self.label, \
               'DecisionTree needs a data set and list of labels.'
        assert ratio <= 1, 'Cannot split data more than 100%.'
        self.trainset = set()
        data = list( self.data )
        shuffle( data )
        index = int( ratio*len( data ) )
        self.testset = data[ : index ]
        self.trainset = data[ index : ]
        self.learn( self.trainset )
        self.test( self.testset )
        print( 'Samples in training set: ', len( self.trainset ) )
        print( 'Samples tested         : ', len( self.testset ) )
        print( 'Total samples          : ',
               len( self.trainset ) + len( self.testset ) )
        self.showResult()
        return
        

    def plot( self, title : str = '' ) -> None :
        self.__visualize( self.tree )
        plt.title(
            title,
            loc = 'center',
            pad = 20,
            bbox = dict(
                facecolor = 'white',
                edgecolor = 'none',
                boxstyle  = 'round'
                )
            )
        plt.axis( 'off' )
        plt.grid( b = None )
        plt.show()
        plt.clf()
        return

    def __plotNode(
        self, x : float, y : float, label : str
        ) -> None :
        plt.text(
            x, y,
            label,
            color = 'black',
            fontsize = 12,
            bbox = dict(
                facecolor = 'white',
                edgecolor = 'black',
                boxstyle  = 'round'
                ),
            verticalalignment = 'center',
            horizontalalignment = 'center'
            )
        return
    
    def __plotBranch(
        self,
        xp : float, xc : float,
        yp : float, yc : float
        ) -> None :
        plt.plot(
            ( xp, xc ),
            ( yp, yc ),
            color = 'black'
            )
        return 
    
    def __visualize(
        self, tree : 'Nested Dict',
        xp : float = 0, yp : float = 0,
        spread : int = 1
        ) -> None :
        if tree :
            child_nm = 0
            siblings = len( tree )
            spread *= siblings
            for node in tree :
                if siblings % 2 == 0 :
                    xc = child_nm + 0.5 - siblings / 2
                else :
                    xc = child_nm - siblings // 2
                xc = xp + xc / spread
                yc = yp - 1
                self.__plotNode( xc, yc, str( node ) )
                if yp != 0 :
                    self.__plotBranch( xp, xc, yp, yc )
                self.__visualize(
                    tree[ node ],
                    xc, yc, spread
                    )
                child_nm += 1
        return
