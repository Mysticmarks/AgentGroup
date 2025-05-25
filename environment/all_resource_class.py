from .resource_class import Resource
import os

class AllResource:
    def __init__(self, save_folder=None)->None:
        '''
        Initializes the AllResource class.
        Input:
            save_folder: str
        Output:
            None
        '''
        self.resource_dict = {}
        self.resource_list = []
        self.initialize(save_folder)

    def initialize(self, save_folder)->int:
        '''
        Reads AllResource information from save_folder.
        Input:
            save_folder: str
        Output:
            success_number: int, how many resources were successfully read.
        '''
        success_number = 0
        for file in os.listdir(save_folder):
            resource_file = os.path.join(save_folder, file)
            resource = Resource(saved_folder=resource_file)
            self.append(resource)
            success_number += 1
        return success_number

    def append(self, new_resource: Resource)->None:
        '''
        Adds new_resource to AllResource.
        Input:
            new_resource: Resource,
        Output:
            None
        '''
        self.resource_dict[new_resource.id_number] = new_resource
        self.resource_list.append(new_resource)

    def get_resource_by_id_number(self, id_number)->Resource:
        '''
        Gets a resource by its ID Number.
        Input:
            id_number: str
        Output:
            resource: Resource
        '''
        return self.resource_dict[id_number]

    def get_all_resource(self)->list:
        '''
        Gets the list of all resources.
        Input:
            None
        Output:
            resource_list: list, [Resource]
        '''
        return self.resource_list

    def get_description(self)->str:
        '''
        Gets the description of all resources.
        Input:
            None
        Output:
            description: str, description of all resources.
        '''
        description = ''
        for resource in self.get_all_resource():
            description += resource.get_description() + '\n'
        return description.strip()