import numpy as np
from JobOffer import JobOffer
from User import User
from sklearn.preprocessing import OneHotEncoder

class Environment():
    """
    Environment.
    """
    def __init__(self, schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places,
                 max_day_step = 10, noise_user = 0, noise_offer = 0, n_users = 30, n_offers = 30, evolutive = False):
        """
        Initialize an instance of the Environment class.

        Arguments
        ---------
        schools_dictionnary: dict(str: School)
            dictionnary to map the name of a school to its instance.
        domains_to_skills_dictionnary: dict(str: list(str))
            map from domains to skills.
        companies: list(str)
            list of the companies.
        hierarchical_levels: list(str)
            list of the different hierarchical levels.
        skills: list(str)
            list of the skills.
        places: list(str)
            List of the places.
        max_day_step: int
            number of steps for a day.
        noise_user: int
            size of the noise for the code of the users
        noise_offer:
            size of the noise for the code of the offers
        n_users: int
            number of users at the beginning.
        n_offers: int
            number of offers at the beginning.
        """

        self.schools_dictionnary  = schools_dictionnary
        self.companies            = companies
        self.hierarchical_levels  = hierarchical_levels
        self.skills               = skills
        self.places               = places

        self.schools_idx   = {school: i for i, school in enumerate(schools_dictionnary)}
        self.companies_idx = {company: i for i, company in enumerate(companies)}

        self.companies_oh = OneHotEncoder(sparse=False).fit(np.asarray(companies).reshape(-1, 1))
        self.skills_oh    = OneHotEncoder(sparse=False).fit(np.asarray(skills).reshape(-1, 1))
        self.places_oh    = OneHotEncoder(sparse=False).fit(np.asarray(places).reshape(-1, 1))
        self.domains_oh   = OneHotEncoder(sparse=False).fit(np.asarray(list(domains_to_skills_dictionnary.keys())).reshape(-1, 1))

        self.domains_to_skills_dictionnary = domains_to_skills_dictionnary  

        self.n_users_0  = n_users
        self.n_offers_0 = n_offers

        self.noise_user  = noise_user
        self.noise_offer = noise_offer

        self.size_embedded_user  = len(skills) + len(places) + self.noise_user
        self.size_embedded_offer = len(skills) + len(places) + 2 + self.noise_offer

        self.evolutive = evolutive

        self.max_day_step = max_day_step

    def reset(self, seed = None):
        """
        Reset the environment.

        Arguments
        ---------
        seed: None or int
            seed for the rng.

        Return
        ---------
        self.state: list()
            state of the environement.
        """      
        self._rng     = np.random.RandomState(seed)

        self.t        = 0
        self.day_step = 0

        self.n_users  = self.n_users_0
        self.n_offers = self.n_offers_0
        
        self.users    = [User.create_user(self.schools_dictionnary, self.domains_to_skills_dictionnary,\
                                          self.companies, self.places, self.skills_oh, self.places_oh,\
                                          self.domains_oh, self._rng, i)\
                         for i in range (self.n_users)]

        self.offers   = [JobOffer.create_job_offer(self.domains_to_skills_dictionnary,\
                                                   self.hierarchical_levels, self.companies,\
                                                   self.places, self.skills_oh, self.places_oh,\
                                                   self.domains_oh, self._rng, i) 
                         for i in range (self.n_offers)]

        self._get_max_days_and_nb_candidates()

        self.network  = self._generate_network(self.users, self.schools_idx, self.companies_idx)

        self._next_state()

        return self.state


    def step(self, action):
        """ 
        Play an action.

        Arguments
        ---------
        action: list()
            the list of the ordinate offers chosen by the agent.

        Return
        --------
        self.state: list()
            state of the environment.
        rewards: np.ndarray
            the list of rewards related to the action.
        clicks: np.ndarray
            the list of clicks (1 for click 0 for no click for each offer in action).
        best_rewards: np.ndarray
            the list of best rewards.
        """

        potential_rewards = np.asarray([self._get_reward(offer) for offer in self.offers])
                                        
        best_rewards = np.sort(potential_rewards)[-len(action):][::-1]

        rewards = potential_rewards[action]

        clicks = []
        for offer_idx, reward in zip(action, rewards):
            click = self._rng.choice([0, 1], p = [1 - reward, reward])
            if click:
                if offer_idx not in self.active_user.clicks:
                    self.offers[offer_idx].nb_candidates += 1

                self.active_user.click(self.offers[offer_idx]._id, self.t)
                

        clicks  = [self._rng.choice([0, 1], p = [1 - reward, reward]) for reward in rewards]

        self._update_env(action, rewards, clicks)

        self._next_state()

        return self.state, rewards, clicks, best_rewards 


    def _get_max_days_and_nb_candidates(self):
        """
        Retrieve the max days and max number of candidates in the allowed offers.
        """

        self.max_days_posted   = 0
        self.max_nb_candidates = 0

        for offer in self.offers:
            if offer.days_posted > self.max_days_posted:
                self.max_days_posted = offer.days_posted

            if offer.nb_candidates > self.max_nb_candidates:
                self.max_nb_candidates = offer.nb_candidates


    def _match(self, u, v):
        """
        Compute the match between two vectors composed of zeros and ones.

        Arguments
        ---------
        u: np.ndarray
            vector u
        v: np.ndarray
            vector v

        Return
        ---------
        distance: int
            distance between u and v based on card(u and v) / card(u or v).
        """

        u_and_v  = u * v
        u_or_v   = u + v - u_and_v
        distance = np.sum(u_and_v) / np.sum(u_or_v)

        return distance

    def _get_reward(self, offer):
        """
        Compute the reward for an offer and the active user.

        Arguments
        ---------
        offer: JobOffer
            the offer.

        Return
        ---------
        reward: int
            the reward.
        """

        match_skills  = self._match(self.active_user.oh_skills, offer.oh_skills)
        match_domains = self._match(self.active_user.oh_domains, offer.oh_domains)
        match_places  = self._match(self.active_user.oh_place, offer.oh_place)

        normalized_days                  = offer.days_posted / self.max_days_posted
        normalized_nb_candidates         = offer.nb_candidates / self.max_nb_candidates
        harmonic_mean_days_nb_candidates = 2 * normalized_days * normalized_nb_candidates / (normalized_days + normalized_nb_candidates)

        days_clicked        = self.t - self.active_user.clicks[offer._id] if offer._id in self.active_user.clicks else np.inf

        weight_days_clicked = 0.5 if days_clicked == 0 else 1 / np.sqrt(days_clicked)


        reward = 5 * match_skills + 3 * harmonic_mean_days_nb_candidates + match_domains + 2 * match_places + 1.5 * weight_days_clicked
        reward /= 12.5

        return reward

    def _generate_network(self, users, schools_idx, companies_idx):
        """
        Generate a network between the schools and companies. If a user has been in a school and is in a company it adds one to the related entry in the network functionning as a matrix.

        Arguments
        ---------
        users: list(User)
            the users
        schools_idx: dict(str: int)
            map the name of schools to an index.
        companies_idx: dict(str: int)
            map the name of the companies to an index.

        Return
        --------
        network: np.ndarray
            the network.
        """

        network= np.zeros((len(companies_idx), len(schools_idx)))

        for user in users:
            for school in user.schools:
                network[companies_idx[user.company], schools_idx[school]] += 1

        return network

    def _update_env(self, id_offers, rewards, clicks):
        """
        Update the environment based on the clicks on the offers by the user.

        Argument
        --------
        id_offers: list(int)
            the index of the offers
        rewards: np.ndarray
            the rewards related to the offers
        clicks: np.ndarray
            the clicks on the offers.
        """


        for id_offer, reward, click in zip(id_offers, rewards, clicks) :
            if click:
                offer = self.offers[id_offer]

                alumnies_in_companies = np.sum([self.network[self.companies_idx[offer.company]][self.schools_idx[school]] for school in self.active_user.schools])
                
                proba_accept = 5 * reward ** 2 + 1 * (1 - 1/alumnies_in_companies) if alumnies_in_companies >= 1 else 5 * reward ** 2

                proba_accept /= 6

                accept = self._rng.choice([1, 0], p = [proba_accept, 1 - proba_accept])

                if accept:
                    for school in self.active_user.schools:
                        self.network[self.companies_idx[self.active_user.company]][self.schools_idx[school]] -= 1
                        self.network[self.companies_idx[offer.company]][self.schools_idx[school]]            += 1
                   
                    if self.evolutive:
                        self.n_users  += 1
                        self.n_offers += 1
                        self.users.remove(self.active_user)
                        self.offers.remove(offer)

                        self.offers.append(JobOffer.create_job_offer(self.domains_to_skills_dictionnary,\
                                                                    self.hierarchical_levels, self.companies,\
                                                                    self.places, self.skills_oh, self.places_oh,\
                                                                    self.domains_oh, self._rng, self.n_offers))
                        new_user = User.create_user(self.schools_dictionnary, self.domains_to_skills_dictionnary,\
                                                    self.companies, self.places, self.skills_oh, self.places_oh,\
                                                    self.domains_oh, self._rng, self.n_users)
                        
                        for school in new_user.schools:
                            self.network[self.companies_idx[new_user.company], self.schools_idx[school]] += 1

                        self.users.append(new_user)

                    break

        if self.day_step == self.max_day_step - 1:
            self.t += 1
            self.day_step = 0

        else:
            self.day_step += 1

        self._get_max_days_and_nb_candidates()



    def _next_state(self):
        """
        Compute the next state
        """

        self.active_user = self._rng.choice(self.users)

        self.state = []

        for idx, offer in enumerate(self.offers):
            self.state.append([self.active_user._id, idx, self._encode(self.active_user, offer)])


    def _encode(self, user, offer):
        """
        Encode a user and an offer.

        Arguments
        ---------
        user: User
            the user.
        offer: JobOffer
            the offer.

        Return
        --------
        code: np.ndarray
            the code.
        """

        user_encoded  = user.code
        offer_encoded = offer.code

        offer_encoded[-2] /= self.max_days_posted
        offer_encoded[-1] /= self.max_nb_candidates

        if self.noise_user > 0:
            noise = self._rng.normal(loc=np.ones(self.noise_user),
                                    scale=np.ones(self.noise_user),
                                    size=self.noise_user)
            user_encoded = np.concatenate([user_encoded, noise])
            
        if self.noise_offer > 0:
            noise = self._rng.normal(loc=np.ones(self.noise_offer),
                                    scale=np.ones(self.noise_offer),
                                    size=self.noise_offer)
            offer_encoded = np.concatenate([offer_encoded[:, :-2], noise, offer_encoded[:, -2:]])

        code = np.concatenate([user_encoded, offer_encoded])

        return code