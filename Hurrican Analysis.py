# names of hurricanes
names = ['Cuba I', 'San Felipe II Okeechobee', 'Bahamas', 'Cuba II', 'CubaBrownsville', 'Tampico', 'Labor Day', 'New England', 'Carol', 'Janet', 'Carla', 'Hattie', 'Beulah', 'Camille', 'Edith', 'Anita', 'David', 'Allen', 'Gilbert', 'Hugo', 'Andrew', 'Mitch', 'Isabel', 'Ivan', 'Emily', 'Katrina', 'Rita', 'Wilma', 'Dean', 'Felix', 'Matthew', 'Irma', 'Maria', 'Michael']

# months of hurricanes
months = ['October', 'September', 'September', 'November', 'August', 'September', 'September', 'September', 'September', 'September', 'September', 'October', 'September', 'August', 'September', 'September', 'August', 'August', 'September', 'September', 'August', 'October', 'September', 'September', 'July', 'August', 'September', 'October', 'August', 'September', 'October', 'September', 'September', 'October']

# years of hurricanes
years = [1924, 1928, 1932, 1932, 1933, 1933, 1935, 1938, 1953, 1955, 1961, 1961, 1967, 1969, 1971, 1977, 1979, 1980, 1988, 1989, 1992, 1998, 2003, 2004, 2005, 2005, 2005, 2005, 2007, 2007, 2016, 2017, 2017, 2018]

# maximum sustained winds (mph) of hurricanes
max_sustained_winds = [165, 160, 160, 175, 160, 160, 185, 160, 160, 175, 175, 160, 160, 175, 160, 175, 175, 190, 185, 160, 175, 180, 165, 165, 160, 175, 180, 185, 175, 175, 165, 180, 175, 160]

# areas affected by each hurricane
areas_affected = [['Central America', 'Mexico', 'Cuba', 'Florida', 'The Bahamas'], ['Lesser Antilles', 'The Bahamas', 'United States East Coast', 'Atlantic Canada'], ['The Bahamas', 'Northeastern United States'], ['Lesser Antilles', 'Jamaica', 'Cayman Islands', 'Cuba', 'The Bahamas', 'Bermuda'], ['The Bahamas', 'Cuba', 'Florida', 'Texas', 'Tamaulipas'], ['Jamaica', 'Yucatn Peninsula'], ['The Bahamas', 'Florida', 'Georgia', 'The Carolinas', 'Virginia'], ['Southeastern United States', 'Northeastern United States', 'Southwestern Quebec'], ['Bermuda', 'New England', 'Atlantic Canada'], ['Lesser Antilles', 'Central America'], ['Texas', 'Louisiana', 'Midwestern United States'], ['Central America'], ['The Caribbean', 'Mexico', 'Texas'], ['Cuba', 'United States Gulf Coast'], ['The Caribbean', 'Central America', 'Mexico', 'United States Gulf Coast'], ['Mexico'], ['The Caribbean', 'United States East coast'], ['The Caribbean', 'Yucatn Peninsula', 'Mexico', 'South Texas'], ['Jamaica', 'Venezuela', 'Central America', 'Hispaniola', 'Mexico'], ['The Caribbean', 'United States East Coast'], ['The Bahamas', 'Florida', 'United States Gulf Coast'], ['Central America', 'Yucatn Peninsula', 'South Florida'], ['Greater Antilles', 'Bahamas', 'Eastern United States', 'Ontario'], ['The Caribbean', 'Venezuela', 'United States Gulf Coast'], ['Windward Islands', 'Jamaica', 'Mexico', 'Texas'], ['Bahamas', 'United States Gulf Coast'], ['Cuba', 'United States Gulf Coast'], ['Greater Antilles', 'Central America', 'Florida'], ['The Caribbean', 'Central America'], ['Nicaragua', 'Honduras'], ['Antilles', 'Venezuela', 'Colombia', 'United States East Coast', 'Atlantic Canada'], ['Cape Verde', 'The Caribbean', 'British Virgin Islands', 'U.S. Virgin Islands', 'Cuba', 'Florida'], ['Lesser Antilles', 'Virgin Islands', 'Puerto Rico', 'Dominican Republic', 'Turks and Caicos Islands'], ['Central America', 'United States Gulf Coast (especially Florida Panhandle)']]

# damages (USD($)) of hurricanes
damages = ['Damages not recorded', '100M', 'Damages not recorded', '40M', '27.9M', '5M', 'Damages not recorded', '306M', '2M', '65.8M', '326M', '60.3M', '208M', '1.42B', '25.4M', 'Damages not recorded', '1.54B', '1.24B', '7.1B', '10B', '26.5B', '6.2B', '5.37B', '23.3B', '1.01B', '125B', '12B', '29.4B', '1.76B', '720M', '15.1B', '64.8B', '91.6B', '25.1B']

# deaths for each hurricane
deaths = [90,4000,16,3103,179,184,408,682,5,1023,43,319,688,259,37,11,2068,269,318,107,65,19325,51,124,17,1836,125,87,45,133,603,138,3057,74]


# Update Recorded Damages
conversion = {"M": 1000000,
              "B": 1000000000}
update_damages = []
def updated_damages(lst):
  for item in lst:
    if item == "Damages not recorded":
      update_damages.append(item)
    elif item[-1] == "M":
      update_damages.append(float(item.strip("M"))*conversion["M"])
    elif item[-1] == "B":
      update_damages.append(float(item.strip("B"))*conversion["B"])
  return update_damages

print(updated_damages(damages))

# Create and view the hurricanes dictionary
#zipped= zip(names,months,years,max_sustained_winds,areas_affected,update_damages, deaths)
#list_zipped = print(list(zipped))
def hurricane_dict(names,months, years, max_sustained_winds, areas_affected,update_damages,deaths):
  hurricanes = {}
  #num_hurricanes = len(names)
  for i in range(len(names)):
    hurricanes[names[i]] = {"Names": names[i], "Month": months[i], "Year": years[i], "Max Sustained Winds": max_sustained_winds[i], "Areas Affected": areas_affected[i], "Damage": update_damages[i], "Deaths": deaths[i]}
  return hurricanes
  print(len(hurricanes))
hurricanes = hurricane_dict(names,months, years, max_sustained_winds, areas_affected,update_damages,deaths)
#print(hurricanes)


# Organizing by Year
# create a new dictionary of hurricanes with year and key

def years_dict(hurricanes):
  hurricane_years = {}
  for values in hurricanes.values():
    #print(values)
    current_year = values["Year"]
    hurricane_years[current_year] = values
  return hurricane_years

print(years_dict(hurricanes))



# Counting Damaged Areas
def damage_dict(hurricanes):
  count_area = 0
  hurricane_damage = {}
  for values in hurricanes.values():
    #print(values)
    area = values["Areas Affected"]
    for i in area:
      if i not in hurricane_damage:
        #count_area = 1
        hurricane_damage[i] = 1
      else:
        #count_area += 1
        hurricane_damage[i] += 1


  return hurricane_damage
#create dictionary of areas to store the number of hurricanes involved in
hurricane_damage = damage_dict(hurricanes)
print(hurricane_damage)


#Calculating Maximum Hurricane Count
def max_damage(lst):
  most_hit = ()
  greatest= 0
  for key,value in hurricane_damage.items():
    #most_hit = list((key,value))
    #print(most_hit)
    for i in key,value:
      #print(value)
      if int(value) < greatest:
        greatest = greatest
      else:
        greatest = value
    return key, greatest

#finding most frequently affected area and the number of hurricanes involved in
print(max_damage(hurricane_damage))

# Calculating the Deadliest Hurricane
def hurricane_mortality(hurricanes):
  max_mortality= []
  max_death = 0
  max_name = ""
  for value in hurricanes.values():
    #print(value)
    deaths = value["Deaths"]
    Name = value["Names"]
    #print(deaths,Name)
    max_mortality.append(Name)
    max_mortality.append(deaths)
    if int(deaths) > max_death:
      max_death = deaths
      max_name = Name
    #else:
      #max_death = deaths
      #max_name = Name
  return max_name, max_death
  #print(max_mortality)
  #print(max_death)

#find highest mortality hurricane and the number of deaths
print(hurricane_mortality(hurricanes))

# Rating Hurricanes by Mortality
mortality_scale = {0: 0,
                   1: 100,
                   2: 500,
                   3: 1000,
                   4: 10000}
def mortality(hurricanes):
  hurricane_mortality = {0:[],1:[],2:[],3:[],4:[],5:[]}
  hurr_keys = list(mortality_scale.keys())
  #print(hurr_keys)
  for value in hurricanes.values():
    death = value["Deaths"]
    name = value["Names"]
    if int(death) == 0 :
      hurricane_mortality[0].append(name)
    elif int(death) < 100:
       hurricane_mortality[1].append(name)
    elif int(death) > 100 and int(death) < 500:
       hurricane_mortality[2].append(name)
    elif int(death) > 500 and int(death) < 1000:
       hurricane_mortality[3].append(name)
    elif int(death) > 1000 and int(death) < 10000:
      hurricane_mortality[4].append(name)
    elif int(death) >1001:
      hurricane_mortality[5].append(name)
  return hurricane_mortality
# categorize hurricanes in new dictionary with mortality severity as key
print(mortality(hurricanes))

#Calculating Hurricane Maximum Damage
def max_damage(hurricanes):
  maximum_damage = 0
  maximum_name = ""
  for value in hurricanes.values():
    damage = value["Damage"]
    name = value["Names"]
    #print(name,damage)
    if damage == "Damages not recorded": continue
    if int(damage) > maximum_damage:
      maximum_damage = damage
      maximum_name = name
  return maximum_name, maximum_damage
# find highest damage inducing hurricane and its total cost
print(max_damage(hurricanes))

# Rating Hurricanes by Damage
damage_scale = {0: 0,
                1: 100000000,
                2: 1000000000,
                3: 10000000000,
                4: 50000000000}

# categorize hurricanes in new dictionary with damage severity as key
def hurr_rating(lst):
  hurricane_scale = {0:[],1:[],2:[],3:[],4:[],5:[]}
  #hurr_keys = list(damage_scale.keys())
  for value in hurricanes.values():
    damages = value["Damage"]
    names = value["Names"]
    if damages == "Damages not recorded": continue
    elif int(damages) == 0:
      hurricane_scale[0].append(names)
    elif int(damages) > 0 and int(damages) <=100000000:
      hurricane_scale[1].append(names)
    elif int(damages) > 100000000 and int(damages) <=1000000000:
      hurricane_scale[2].append(names)
    elif int(damages) > 1000000000 and int(damages) <=10000000000:
      hurricane_scale[3].append(names)
    elif int(damages) > 10000000000 and int(damages) <=50000000000:
      hurricane_scale[4].append(names)
    elif int(damages) > 50000000001:
      hurricane_scale[5].append(names)
  return hurricane_scale

print(hurr_rating(hurricanes))
