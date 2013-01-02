
import os

class Ensight:
    
    def __init__(self, filename='output', vector_var_names=None,
            scalar_var_names=None):

        directory = './ensight_files/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.__geo_file = open(directory+filename+'.geo','w')
        
        self.__vv_names = vector_var_names
        self.__sv_names = scalar_var_names
        self.__fname = filename
        self.times = []

        if self.__vv_names != None:
            self.__vector_var_files = [ open(directory+afilename+'.var','w') 
                    for afilename in self.__vv_names ]
        
        if self.__sv_names != None:
            self.__scalar_var_files = [ open(directory+afilename+'.var','w') 
                    for afilename in self.__sv_names ]

        return


    def write_case_file(self):
        """Initialize Ensight case file"""
        
        directory = './ensight_files/'
        self.__case_file = open(directory+self.__fname+'.case','w')

        print >> self.__case_file, 'FORMAT'
        print >> self.__case_file, 'type: ensight gold'
        print >> self.__case_file, 'GEOMETRY'
        print >> self.__case_file, 'model: 1 1 ' + self.__fname + '.geo'
        print >> self.__case_file, 'VARIABLE'

        if self.__vv_names != None:
            for item in self.__vv_names:
                print >> self.__case_file, ('vector per node: 1 1 ' + 
                        item + ' ' + item +'.var')

        if self.__sv_names != None:
            for item in self.__sv_names:
                print >> self.__case_file, ('scalar per node: 1 1 ' + 
                        item + ' ' + item +'.var')

        print >> self.__case_file, 'TIME'
        print >> self.__case_file, 'time set: 1'
        print >> self.__case_file, 'number of steps: ' + str(len(self.times))
        print >> self.__case_file, 'time values: '
        for item in self.times:
            print >> self.__case_file, item
        print >> self.__case_file, 'FILE'
        print >> self.__case_file, 'file set: 1'
        print >> self.__case_file, 'number of steps: ' + str(len(self.times))

        self.__case_file.close()

        return

    #Create Ensight Format Case file
    def write_geometry_file_time_step(self, x, y):
        """ Initialize Ensight geometry file"""

        print >> self.__geo_file, 'BEGIN TIME STEP'
        print >> self.__geo_file, 'Ensight Gold geometry file\n'
        print >> self.__geo_file, 'node id off'
        print >> self.__geo_file, 'element id off'
        print >> self.__geo_file, 'part'
        print >> self.__geo_file, '1'
        print >> self.__geo_file, 'grid'
        print >> self.__geo_file, 'coordinates'
        print >> self.__geo_file, len(x)
        for item in x:
            print >> self.__geo_file, item 
        for item in y:
            print >> self.__geo_file, item 
        for item in range(len(x)):
            print >> self.__geo_file, 0.0
        print >> self.__geo_file, 'point'
        print >> self.__geo_file, len(x)
        for item in range(len(x)):
            print >> self.__geo_file, item + 1 
        print >> self.__geo_file, 'END TIME STEP'

        return

    def write_vector_variable_time_step(self, variable_name, variable, time):

        write_index = None
        for index,aname in enumerate(self.__vv_names):
            if variable_name == aname:
                write_index = index
                break

        print >> self.__vector_var_files[write_index], 'BEGIN TIME STEP'
        print >> self.__vector_var_files[write_index], 'time = ', time
        print >> self.__vector_var_files[write_index], 'part'
        print >> self.__vector_var_files[write_index], '1'
        print >> self.__vector_var_files[write_index], 'coordinates'
        for xyz in variable:
            for item in xyz:
                print >> self.__vector_var_files[write_index], item

        print >> self.__vector_var_files[write_index], 'END TIME STEP'

        return
    

    def write_scalar_variable_time_step(self, variable_name, variable, time):

        write_index = None
        for index,aname in enumerate(self.__sv_names):
            if variable_name == aname:
                write_index = index
                break

        print >> self.__scalar_var_files[write_index], 'BEGIN TIME STEP'
        print >> self.__scalar_var_files[write_index], 'time = ', time
        print >> self.__scalar_var_files[write_index], 'part'
        print >> self.__scalar_var_files[write_index], '1'
        print >> self.__scalar_var_files[write_index], 'coordinates'
        for item in variable:
            print >> self.__scalar_var_files[write_index], item

        print >> self.__scalar_var_files[write_index], 'END TIME STEP'

        return


    def append_time_step(self,time):

        self.times.append(time)

        return 


    def finalize(self):
        
        self.__geo_file.close()
        
        if self.__vv_names != None:
            for item in self.__vector_var_files:
                item.close()
        
        if self.__sv_names != None:
            for item in self.__scalar_var_files:
                item.close()

        return



