import os


class Cleanser:
    def __init__(self, data):
        self.data = data
        self.cleanse_authors()

    def cleanse_authors(self):
        cleansed_authors = self.data['authors']
        cleansed_authors = cleansed_authors.replace(to_replace=r'Contributor', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'\n', value=' ', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r',[ ]+$', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'["\']', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'\/', value=' ', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'-', value=' ', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r';', value=',', regex=True)

        cleansed_authors = cleansed_authors.replace(to_replace=r'Ph\.D\.[,]?', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'Jr\.[,]?', value='', regex=True)

        cleansed_authors = cleansed_authors.replace(to_replace=r'[ ]+', value=' ', regex=True)

        self.data['authors'] = cleansed_authors
        #self.data['authors'].to_csv(path=os.path.join("resources", "cleansed_authors.csv"), header=True, mode='w')