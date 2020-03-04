class School():
    """
    School.
    """
    def __init__(self, name = "", place = "", domains = []):
        """
        Initalize an instance of the School class.

        Arguments
        ---------
        name: str
            name of the school.
        place: str
            place of the school.
        domains: list(str)
            list of the domains of the school.
        """
        self.name     = name
        self.domains  = domains
        self.place    = place
    
    def __str__(self):
        return self.name