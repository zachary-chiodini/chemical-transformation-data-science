from math import log2
from typing import Any, Set, Tuple

def subset(
    S : Set[ Tuple[ Any ] ],
    x : Any, # value
    j : int  # attribute index
    ) -> Set[ Tuple[ Any ] ] :
    '''
    Subset of S where x is in each tuple Si
    '''
    return { Si for Si in S if Si[ j ] == x }

def majorityVote(
    S : Set[ Tuple[ Any ] ]
    ) -> Any :
    count = {}
    for Si in S :
        yi = Si[ -1 ]
        if yi in count :
            count[ yi ] += 1
        else :
            count[ yi ] = 1
    return max( count )

def split(
    S : Set[ Tuple[ Any ] ],
    j : int # attribute index
    ) -> Set[ Any ] :
    '''
    Split attribute j
    '''
    return { Si[ j ] for Si in S }

def infoEntropy(
    S : Set[ Tuple[ Any ] ]
    ) -> float :
    '''
    Information entropy of the set S
    '''
    prob = {}
    for Si in S :
        yi = Si[ - 1 ]
        if yi in prob :
            prob[ yi ] += 1 / len( S )
        else :
            prob[ yi ] = 1 / len( S )
    return -sum( p*log2( p ) for p in prob.values() )

def gainRatio(
    S : Set[ Tuple[ Any ] ],
    j : int # attribute index
    ) -> float :
    '''
    Information gain ratio of an attribute j
    '''
    IG = infoEntropy( S ) # information gain
    IV = 0 # instrinsic value
    for x in split( S, j ) :
        Sx = subset( S, x, j )
        IG -= len( Sx )*infoEntropy( Sx ) / len( S )
        IV -= len( Sx )*log2( len( Sx ) / len( S ) ) / len( S )
    return IG / IV if IV else 0

def bestAttribute(
    S : Set[ Tuple[ Any ] ],
    n : int # number of attributes
    ) -> int :
    best_j, maxIGR = 0, 0.0
    for j in range( n ) :
        IGR = gainRatio( S, j )
        if IGR > maxIGR :
            best_j, maxIGR = j, IGR
    return best_j
