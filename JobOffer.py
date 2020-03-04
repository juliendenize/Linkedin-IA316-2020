import numpy as np

class JobOffer():
    """
    Job offer.
    """
    def __init__(self, skills_oh, places_oh, domains_oh, company = "", place = "", hierarchical_level = "",
                domains = [], days_posted = 0, nb_candidates = 0, skills = [], _id = 0):

        """
        Initalize an instance of the JobOffer class.

        Arguments
        ---------
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the user.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.
        domains_oh: OneHotEncoder
            one hot encoder fitted on domains to encode the domains.
        company: str
            company name of the job offer.
        place: str
            place of the job offer.
        hierarchical_level: str
            hierarchical level of the job offer.
        domains: list(str)
            domains of the offer.
        days_posted: int
            number of days since the offer is posted.
        nb_candidates: int
            number of candidates on the offer.
        skills: list(str)
            skills required by the job offer.
        _id: int
          id of the offer.
        """

        self._id                = _id
        self.company            = company
        self.place              = place
        self.days_posted        = days_posted
        self.hierarchical_level = hierarchical_level
        self.domains            = domains
        self.skills             = skills
        self.nb_candidates      = nb_candidates
        self.oh_domains         = domains_oh.transform(np.asarray([domain for domain in self.domains]).reshape(-1, 1)).sum(axis = 0)
        self._encode_offer(skills_oh, places_oh)

    def _encode_offer(self, skills_oh, places_oh):
        """
        Encode the job offer in number format to give features to agents.

        Arguments
        ---------
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the skills.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.

        Return
        ---------
        code: np.ndarray
            the encoded job offer.
        """
        self.oh_skills = skills_oh.transform(np.asarray(self.skills).reshape(-1, 1)).sum(axis = 0)
        self.oh_place  = places_oh.transform(np.asarray(self.place).reshape(-1, 1)).sum(axis = 0)
        self.code      = np.concatenate([self.oh_skills, self.oh_place, np.asarray([self.days_posted, self.nb_candidates])])

    @staticmethod
    def create_job_offer(domains_to_skills_dictionnary, hierarchical_levels, companies, places, skills_oh, places_oh, domains_oh, rng, _id):
        """
        Create a user.

        Arguments
        ---------
        domains_to_skills_dictionnary: dict(str: list(str))
            map from domains to skills.
        hierarchical_levels: list(str)
            list of the different hierarchical levels.
        companies: list(str)
            list of the companies.
        places: list(str)
            list of the places.
        skills_oh: OneHotEncoder
            one hot encoder fitted on skills to encode the skills.
        places_oh: OneHotEncoder
            one hot encoder fitted on places to encode the places.
        domains_oh: OneHotEncoder
            one hot encoder fitted on domains to encode the domains.
        rng: rng
            rng from numpy to obtain reproducible results.
        _id: int
            id of the offer.

        Return
        ---------
        offer: JobOffer
            the created offer.
        """
        hierarchical_level = rng.choice(hierarchical_levels)

        domains    = rng.choice(list(domains_to_skills_dictionnary.keys()), rng.choice([1, 2], p = [0.95, 0.05]), replace = False) 

        min_skills = 1.5 if hierarchical_level == "intern" else 2.5 if hierarchical_level == "junior" else 3.5

        nb_skills  = np.round(rng.exponential(0.6 * len(domains)) + min_skills)

        skills     = rng.choice([skill for domain in domains for skill in domains_to_skills_dictionnary[domain]], min(int(nb_skills), (len(domains) - 1) * 3 + 5), replace=False)

        company = rng.choice(companies)
        place   = rng.choice(places)

        days_posted = np.round(rng.exponential(7)) + 1

        nb_candidates = np.round(np.random.exponential(days_posted * 1.2))

        offer = JobOffer(skills_oh, places_oh, domains_oh, company = company, place = place, hierarchical_level = hierarchical_level,
                         domains = domains, days_posted = days_posted, nb_candidates = nb_candidates, skills = skills, _id = _id)

        return offer

    def __str__(self):
        return f"Id: {self._id}\nCompany: {self.company}\nPlace: {self.place}\nHierarchical_level: {self.hierarchical_level}\nDomains: {self.domains}\nDays_posted: {self.days_posted}\nNb_candidates: {self.nb_candidates}\nSkills: {self.skills}\nOh_skills: {self.oh_skills}\n"
