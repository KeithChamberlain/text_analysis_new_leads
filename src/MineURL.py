import numpy as np

class MineURL:
    def __init__(self, full_url):
        '''
        Initializes the MineURL class. Given a full URL, the URL 
        is decomposed into the components: base URL, extended text
        and search string, upon assignment.
        '''
        url_dictionary = dict()
        self.url_dictionary = url_dictionary
        self.url_dictionary['full_url'] = full_url if not full_url in ["", np.nan] else np.nan
        self.url_dictionary["protocols"] = self.get_protocols()
        self.url_dictionary["base_url"] = self.get_base_url(protocol = self.find_base_protocol())
        self.url_dictionary["extended_text"] = np.nan
        self.url_dictionary["forward_slash"] = self.get_element()
        self.url_dictionary["search_string"] = self.clean_substring(self.get_search_string())
        self.url_dictionary["extended_text"] = self.get_extended_url()
    
    
    
    def check_url(self, url):
        ''' 
        check_url checks to see if the URL is a string
        and returns True, else False.
        '''
        if type(url) != str:
            return False
        else:
            return True
    
    
    
    def find_all(self, a_str, sub, overlapping_matches = False):
        '''
        Edited version of Function provided by Pratik Deoghare on Stack Exchange. 
        https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
        
        Function: find_all takes a string (a_str), a substring (sub) and returns the indexes
        of all of the occurences of sub in a_str. 
        '''
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            # use start += 1 to find overlapping matches
            start += 1 if overlapping_matches else len(sub) 



    def find_base_protocol(self):
        '''
        find_base_protocol finds the portion of the URL that 
        contains the base protocol. Returns the first 
        instance of the protocol.
        '''
        url = self.url_dictionary["full_url"]
        if self.check_url(url):
            for key, lst in self.url_dictionary["protocols"].items():
                if 0 in lst:
                    return key
        else:
            return np.nan



    def get_protocols(self, protocol_list=["https", "http", "ftp"]):
        '''
        get_protocols gets a list of protocols present in the URL
        protocol_list: a list of protocols to check for.
        return: the list of protocols or np.nan
        '''
        url = self.url_dictionary["full_url"]
        if self.check_url(url):
            protocols = dict()
            for proto in protocol_list:
                nth_time = 0
                index = np.nan
                if ((proto in url) and (not [proto in key for key in protocols.keys()])):
                    protocols[proto] = list()
                    while index != -1:
                        last_index = index
                        index = url.find(proto, nth_time)
                        if last_index != index:
                            if index != -1:
                                protocols[proto].extend([index])
                        nth_time += 1
            return protocols
        else: 
            return np.nan



    def get_element(self, element = {"forward_slash": ["/"]}):
        '''
        get_element gets the specified element in 
        the search dictionary and returns the location.
        '''
        url = self.url_dictionary["full_url"]
        if self.check_url(url):
            item = list(element.values())[0][0]
            name = list(element.keys())[0]
            result = list(self.find_all(url, item))
            element[name] = [item]
            element[name].append(result)
            return element[name]
        else:
            return np.nan



    def get_base_url(self, protocol = "https"):
        '''
        get_bae_url gets the base URL from the right side
        of the protocol to the (possibly) first forward slash
        '''
        full_url = self.url_dictionary["full_url"]
        if self.check_url(full_url):
            if type(protocol) == str:
                protocol = protocol + "://"
                length_base_protocol = len(protocol)
                url = full_url[length_base_protocol:]
                first_slash = url.find("/")
                if first_slash == -1:
                    return url
                else:
                    return url[:first_slash]
            else:
                return np.nan
        else:
            return np.nan




    def clean_substring(self, obj, 
        delimiters = ["%20", "%2f", "%27", "+", "-", "_", "//", "/", "https","http","ftp","%3a", ":", ";", ".", "q=", "=", "?"],
        replacement = [" ", " ", "", " ", " ", " ", " ", " ", "","","",""," "," "," "," "," ", " "]):
        '''
        clean_substring performs some cleaning of the 
        string. This is used for non base_url portions of the 
        URL. 
        given delimiters, replace the appropriate value. 
        '''
        if self.check_url(obj):
            for string, replace in zip(delimiters, replacement):
                if ((type(obj) != str) or (type(string) != str) or (type(replace) != str)):
                    continue
                else:
                    obj = obj.replace(string, replace)
            return obj
        else:
            return np.nan



    def get_search_string(self, 
        tokens = {"keys": ["search?","maps?","serp?", "RU"], "start_token":"q=", "stop_token":"&"}):
        '''
        get_search_string searches the URL for the appriate substring
        '''
        url = self.url_dictionary["full_url"]
        if self.check_url(url):
            for key in tokens["keys"]:
                if key in url:
                    loc = self.url_dictionary["forward_slash"][1][2]
                    start_key_loc = url[loc:].find(key) # first occurence in extended url
                    start_token_loc = url[loc:][start_key_loc:].find(tokens["start_token"])+1
                    stop_token_loc = url[loc:][start_key_loc:][start_token_loc:].find(tokens["stop_token"])+1

                    return url[loc:][(start_token_loc+len(tokens["start_token"])):(start_token_loc+stop_token_loc)]
            return np.nan
        else:
            return np.nan




    def get_extended_url(self):
        '''
        get_extended_url returns the extended URL from the URL if
        it exists, and np.nan otherwise. 
        '''
        url = self.url_dictionary['full_url']
        if self.check_url(url):
            uncleaned_search_string = self.get_search_string()
            loc = np.nan
            try:
                loc = self.get_element()[1][2]
                extended_url = url[loc:]    
                delimiters = [uncleaned_search_string, "%20", "%2f", "%27", "%3F", "%3D", "%3a", "q=", 
                "UTF-8", "https", "http", "ftp", "&sourceid", "&source",  "_ylu", "/RE", "/RO","/RU",
                "%3A", "_ylt","120&ved", "11&ved","//", "/", ":", ";", "=", "?", "client", "browser", 
                "firefox", "chrome", "safari","%2F", "web", "urlsa", "&url", "1&ved", "web&cd", "&usg",
                "2&ved","source", "url", "t&rct", "8&ved", "&cd", "mobile", "%2B", "9&ved", "&ved", "&",
                "imgref", "imgres", "search", "_", "+", "-", "www.", ".com%", ".com", ".dmg", ",", ".",
                "    ", "    ", "   ", "  "]
                replacement = ["", " ", " ", "", "", "", " ", " ", 
                " "," ", " "," "," "," "," "," ", " "," ",
                " ", " "," ", " "," "," "," ", " ", " ", " ", " ", " ", 
                " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
                " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
                " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
                " ", " "," "," "," "]
                return self.clean_substring(extended_url, delimiters, replacement)
            except IndexError:
                return np.nan
        else:
            return np.nan


    def get_record(self, full_url):
        '''
        get_record returns the cleaned record containing
        the base URL, extended text, and the search string, or NAN otherwise.
        '''
        record = [self.url_dictionary["base_url"], self.url_dictionary["extended_text"], 
                    self.url_dictionary["search_string"]]
        if full_url:
            record.append(self.url_dictionary["full_url"])
        return record


class ManyURL:
    '''
    class ManyURL deals with the actions associated with many urls at a time.
    '''
    def __init__(self, series):
        ranger = range(len(series))
        n_MineURL = 0
        n_nan = 0
        for row in ranger:
            globals()[f'__MineURL_{row}__'] = MineURL(series[row])
            n_MineURL += 1
        if globals()[f'__MineURL_{row}__'].url_dictionary['full_url'] in ["", np.nan]:
            n_nan += 1
        self.n_MineURL = n_MineURL
        self.n_nan = n_nan
        self.full_text = False
        string = f'''
        {len(series)} record(s) read to make objects; with a total of {n_MineURL} MineURL Objects 
        created, containing {n_nan} nan URL text record(s).
        '''
        print(string)

    def get_records(self, full_text=False):
        '''
        get_records is the MineURL get_record equivalent for many records.
        It returns all records in the catalog. 
        '''
        self.full_text = full_text
        lst = list()
        n_nan_read = 0
        n_MineURL_read = 0
        ranger = range(self.n_MineURL)
        for row in ranger:
            data = globals()[f'__MineURL_{row}__'].get_record(full_text)
            if np.nan in data:
                n_nan_read += 1
            lst.append(data)
            n_MineURL_read += 1
        string = f'''
        Read {n_MineURL_read} MineURL objects,
        and {n_nan_read} records with at least one nan: 
        '''
        print(string)
        return lst




if __name__ == "__main__":
    url_list = ["https://r.search.yahoo.com/_ylt=A0geK.eJAcpgPO0ALiPBGOd_;_ylu=Y29sbwNiZjEEcG9zAzEEdnRpZAMEc2VjA3Ny/RV=2/RE=1623880201/RO=10/RU=https%3a%2f%2fwww.tmjsleepapnea.com%2f/RK=2/RS=ZKpG2Bot17MZo3eh7x5q5wmxg8s-",
    "https://www.google.com/search?client=firefox-b-m&sxsrf=ALeKk00smcDsr5cQMhZYQL8lj_h170GQ8w:1623691430943&q=i%27m+scared+my+snoring+will+scare&spell=1&sa=X&ved=2ahUKEwi2h9z30ZfxAhXK_ioKHXEUA9sQBXoECAwQAQ",
    "https://www.dentistrywithaheart.com/services/","", np.nan, None, "https://www.google.com"]
    one = MineURL(url_list[0])
    two = MineURL(url_list[1])
    three = MineURL(url_list[2])
    four = MineURL(url_list[3])
    print(four.url_dictionary)


    many = ManyURL(url_list)
    print(many.n_nan, many.n_MineURL)
    print(many.get_records())