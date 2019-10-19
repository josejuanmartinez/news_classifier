import os


class Cleanser:
    def __init__(self, data):
        self.data = data
        self.cleanse_link()
        # self.cleanse_authors()
        # self.cleanse_headline()
        # self.cleanse_short_description()

    def cleanse_link(self):
        cleansed_link = self.data['link']
        cleansed_link = cleansed_link.replace(to_replace=r'https://www.huffingtonpost.com/entry/', value='', regex=True)
        cleansed_link = cleansed_link.replace(to_replace=r'.htm[l]?', value='', regex=True)
        cleansed_link = cleansed_link.replace(to_replace=r'[-_]', value=' ', regex=True)
        cleansed_link = cleansed_link.replace(to_replace=r'5b[\w]+', value='', regex=True)
        #cleansed_link = cleansed_link.replace(to_replace=r'[0-9]+', value='', regex=True)
        #cleansed_link = cleansed_link.replace(to_replace=r'\b[a-z]{1}\b', value='', regex=True)
        self.data['link'] = cleansed_link
        #print(self.data['link'])

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

        cleansed_authors = cleansed_authors.replace(to_replace=r'["]+', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'[\']+', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'[\']+', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'[\(]+', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'[\)]+', value='', regex=True)
        cleansed_authors = cleansed_authors.replace(to_replace=r'[‘’]+', value='', regex=True)

        self.data['authors'] = cleansed_authors
        #self.data['authors'].to_csv(path=os.path.join("resources", "cleansed_authors.csv"), header=True, mode='w')

    def cleanse_headline(self):
        cleansed_headline = self.data['headline']
        cleansed_headline = cleansed_headline.replace(to_replace=r'["]+', value='', regex=True)
        cleansed_headline = cleansed_headline.replace(to_replace=r'[\']+', value='', regex=True)
        cleansed_headline = cleansed_headline.replace(to_replace=r'[\']+', value='', regex=True)
        cleansed_headline = cleansed_headline.replace(to_replace=r'[\(]+', value='', regex=True)
        cleansed_headline = cleansed_headline.replace(to_replace=r'[\)]+', value='', regex=True)
        cleansed_headline = cleansed_headline.replace(to_replace=r'[‘’]+', value='', regex=True)

        self.data['headline'] = cleansed_headline
        self.data['headline'].to_csv(path=os.path.join("resources", "cleansed_headline.csv"), header=True, mode='w')

    def cleanse_short_description(self):
        cleanse_short_description = self.data['short_description']
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'["]+', value='', regex=True)
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'[\']+', value='', regex=True)
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'[\']+', value='', regex=True)
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'[\(]+', value='', regex=True)
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'[\)]+', value='', regex=True)
        cleanse_short_description = cleanse_short_description.replace(to_replace=r'[‘’]+', value='', regex=True)

        self.data['short_description'] = cleanse_short_description
        self.data['short_description'].to_csv(path=os.path.join("resources", "cleansed_short_desc.csv"), header=True, mode='w')
