import numpy as np

class User():
    def __init__(self, skills_oh, places_oh, domains_oh, schools_dictionnary, skills = [], age = 0, place = "", company = "",
                schools = [], _id = 0):
        """
        Initalize an instance of the User class.

        Arguments
        ---------
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the user.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.
        domains_oh: OneHotEncoder
            one hot encoder fitted on domains to encode the domains.
        schools_dictionnary: dict(str: School)
            dictionnary to map the name of a school to its instance.
        skills: list(str)
            list of the user's skills in string format.
        age: int
            age of the user.
        place: str
            place of the user.
        company: str
            company name of the user.
        _id: int
            id of the user.
        """

        self._id          = _id
        self.skills       = skills
        self.age          = age
        self.place        = place
        self.company      = company
        self.schools      = schools
        self.clicks       = {}
        self.oh_domains   = domains_oh.transform(np.asarray([domain for school in self.schools for domain in schools_dictionnary[school].domains]).reshape(-1, 1)).sum(axis = 0)
        self._encode_user(skills_oh, places_oh)
    
    def _encode_user(self, skills_oh, places_oh):
        """
        Encode the user in number format to give features to agents.

        Arguments
        ---------
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the skills.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.

        Return
        ---------
        code: np.ndarray
            the encoded user.
        """

        self.oh_skills = skills_oh.transform(np.asarray(self.skills).reshape(-1, 1)).sum(axis = 0)
        self.oh_place  = places_oh.transform(np.asarray(self.place).reshape(-1, 1)).sum(axis = 0)

        self.code      = np.concatenate([self.oh_skills, self.oh_place])

    def click(self, offer_id, time):
        """
        Save the click of the user on an offer.

        Arguments
        ---------
        offer_id: int
            id of the offer.
        time: int
            time of the click.
        """

        self.clicks[offer_id] = time

    @staticmethod
    def create_user(schools_dictionnary, domains_to_skills_dictionnary, companies, places, skills_oh, places_oh, domains_oh, rng, _id):
        """
        Create a user.

        Arguments
        ---------
        schools_dictionnary: dict(str: School)
            dictionnary to map the name of a school to its instance.
        domains_to_skills_dictionnary: dict(str: list(str))
            map from domains to skills.
        companies: list(str)
            List of the companies.
        places: list(str)
            List of the places.
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the skills.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.
        domains_oh: OneHotEncoder
            one hot encoder fitted on domains to encode the domains.
        rng: rng
            rng from numpy to obtain reproducible results.
        id: int
           id of the user.

        Return
        ---------
        user: User
            the created user.
        """

        age     = rng.randint(20,60)
        schools = rng.choice(list(schools_dictionnary.keys()), rng.choice([1, 2], p = [0.95, 0.05]), replace = False) 

        available_skills = list(set([skill for school in schools \
                                           for domain in schools_dictionnary[school].domains \
                                           for skill  in domains_to_skills_dictionnary[domain]]))

        expo = np.round(rng.exponential(0.3) * len(schools)) + age // 17 + 1

        nb_skills_to_choose = min(int(expo), 5 + (len(schools) - 1) * 3)

        _skills = rng.choice(available_skills, nb_skills_to_choose, replace = False)

        company = rng.choice(companies)
        place   = rng.choice(places)

        user = User(skills_oh, places_oh, domains_oh, schools_dictionnary, skills = _skills, age = age, place = place, company = company,
                    schools = schools, _id = _id)

        return user

    def __str__(self):
        return (f"Id: {self._id}\nAge: {self.age}\nSchools: {[str(school) for school in self.schools]}\nSkills : {self.skills}\nOh_skills: {self.oh_skills}\nPlace  : {self.place}\n")
