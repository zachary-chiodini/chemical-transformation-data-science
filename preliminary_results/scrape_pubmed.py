import re
from tools import RunTime
from Bio import Entrez, Medline
from typing import Dict, Set, Union

class PubMed :

    PubMedID = Union[ int, str ]
    Abstract = str

    def __init__( self, email = '' ) :
        self.email = email
        self.runtime = RunTime()

    def extract( self, ids : Set[ PubMedID ] ) -> Dict[ PubMedID, Abstract ] :
        result = {}
        self.runtime.reset()
        self.runtime.total = len( ids )
        for pmid in ids :
            self.runtime.log = 'Extracting ID {} ...'.format( pmid )
            self.runtime.progbar()
            result[ pmid ] = Medline.read(
                Entrez.efetch(
                    db = 'pubmed',
                    id = pmid,
                    email = 'chiodini.zachary@epa.gov',
                    retmode = 'text',
                    rettype = 'medline'
                    )
                ).get( 'AB' ) # abstract
        return result
            
