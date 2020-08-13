from data.external_data import CovidData

from pyomo.environ import *
import numpy as np
from pyomo.opt import SolverFactory
import pyomo.environ

class TransOpt:
    
    def __init__(self, data):
        self.data = data
    
    def create_model(self, df_cases, time_horizon=28,start_day=120, n1=20000, n2=300, n3=100, n4=150):
        
        def haversine_np(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)

            All args must be of equal length.    

            """
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

            c = 2 * np.arcsin(np.sqrt(a))
            miles = 3959 * c
            return miles
        
        
        df_geo = self.data.df_geo.copy().reset_index()
        dc_geo = self.data.dc_geo.copy().reset_index()
        production_geo = self.data.production_geo.copy().reset_index()

        temp = dc_geo.assign(A=1).merge(df_geo.assign(A=1), on='A').drop('A', 1)
        temp['distance'] = haversine_np(temp['longitude_x'], temp['latitude_x'], temp['longitude_y'], temp['latitude_y'])
        distance_matrix_demand = temp.set_index(['FIPS_x', 'FIPS_y'])['distance'].to_dict()

        temp = production_geo.assign(A=1).merge(dc_geo.assign(A=1), on='A').drop('A', 1)
        temp['distance'] = haversine_np(temp['longitude_x'], temp['latitude_x'], temp['longitude_y'], temp['latitude_y'])
        distance_matric_production = temp.set_index(['FIPS_x', 'FIPS_y'])['distance'].to_dict()

        temp = dc_geo.assign(A=1).merge(dc_geo.assign(A=1), on='A').drop('A', 1)
        temp['distance'] = haversine_np(temp['longitude_x'], temp['latitude_x'], temp['longitude_y'], temp['latitude_y'])
        distance_matrix_dc = temp.set_index(['FIPS_x', 'FIPS_y'])['distance'].to_dict()

        miles_per_day = 650
        transit_time_dc_dem = {k:int(v/miles_per_day) + 1 for k,v in distance_matrix_demand.items()}
        transit_time_dc_dc = {k:int(v/miles_per_day) + 1 for k,v in distance_matrix_dc.items()}
        transit_time_prod_dc = {k:int(v/miles_per_day) + 1 for k,v in distance_matric_production.items()}
        # Creation of a Concrete Model
        self.model = ConcreteModel()


        holding_capacities = {'47137':160000,'56027':60000, '41033': 25000, '39109': 120000}
        start_capacities = {'47137':n1,'56027':n2, '41033': n3, '39109': n4}
        production_capacity = {'13103':500000,'26119':250000, '38027': 100000}
        shipping_times = None

        self.model.dcs = Set(initialize=holding_capacities.keys(), doc='DC')
        self.model.demand_nodes = Set(initialize=df_geo['FIPS'], doc='Demand_nodes')
        self.model.production_nodes = Set(initialize=production_capacity.keys())
        self.model.t = Set(initialize=range(time_horizon), doc='horizon')

        demand = dict()
        for k,v in df_cases.iloc[start_day:start_day + time_horizon].reset_index().drop(columns = 'date').to_dict().items():
            for kk, vv in v.items():
                demand[k,kk] = vv

        self.model.holding_capacities = Param(self.model.dcs, initialize=holding_capacities, doc='Holding Capacity of DC i in cases per day')
        self.model.production_capacities = Param(self.model.production_nodes, initialize=production_capacity, doc='Holding Capacity of DC i in cases per day')

        self.model.demand = Param(self.model.demand_nodes, self.model.t, initialize=demand, doc='Demand at market j in cases')

        self.model.dist_mat_dem = Param(self.model.dcs,self.model.demand_nodes, initialize=distance_matrix_demand, doc='Distance in miles')
        self.model.dist_mat_dc = Param(self.model.dcs,self.model.dcs, initialize=distance_matrix_dc, doc='Distance in miles from dc to dc')
        self.model.dist_mat_prod_dc = Param(self.model.production_nodes, self.model.dcs, initialize= distance_matric_production, doc = 'distacne in miled to prod plants to DCs')
        #  Scalar f  freight in dollars per case per thousand miles  /90/ ;
        self.model.freight_cost_dc_dem = Param(initialize=0.01, doc='Freight in dollars per case per mile from a dc to a demand node')
        self.model.freight_cost_dc_dc = Param(initialize = 0.002, doc = 'Freight cost from dc to dc in dollars per case per mile')
        self.model.storage_cost = Param(initialize=0.001, doc = 'cost of storing case at dc')
        self.model.production_cost = Param(initialize = 0.009, doc = 'production price per unit')

        def cost_init(model, i, j):
            return model.freight_cost_dc_dem * (model.dist_mat_dem[i,j] + 10.0) / 1000.0

        def cost_dc(model,i,j):
            if i != j:
                val = model.freight_cost_dc_dc* (model.dist_mat_dc[i,j] + 10.0)/1000.0
                return val
            else:
                return 1000.0

        def cost_prod_dc(mode, i ,j):
            return self.model.freight_cost_dc_dc *  (self.model.dist_mat_prod_dc[i,j] + 10.0)/1000.0


        self.model.c = Param(self.model.dcs, self.model.demand_nodes, initialize=cost_init, doc='Transport cost in thousands of dollar per case')
        self.model.c_dc = Param(self.model.dcs, self.model.dcs, initialize=cost_dc, doc = 'transport cost from dc to dc cheaper than other freight cost')
        self.model.c_prod = Param(self.model.production_nodes, self.model.dcs, initialize=cost_prod_dc, doc = 'transport cost from prod to dc cheaper than other freight cost')

        self.model.cases_dc_dem = Var(self.model.dcs, self.model.demand_nodes, self.model.t, bounds=(0.0,None), doc='Shipment quantities in case to demand')
        self.model.cases_held = Var(self.model.dcs, self.model.t, bounds = (0,None), doc = 'cases held at dc')
        self.model.cases_dc_dc = Var(self.model.dcs, self.model.dcs, self.model.t, bounds=(0.0,None), doc = 'shipment quantities in cases between dcs') 
        self.model.cases_produced = Var(self.model.production_nodes, self.model.t, bounds = (0.0,None), doc = 'number of units produced')
        self.model.cases_prod_dc = Var(self.model.production_nodes, self.model.dcs, self.model.t, bounds=(0.0,None), doc = 'shipment quantities in cases between production and dcs') 

        self.model.continuity_cases = ConstraintList()
        for dc in self.model.dcs:
            for t in self.model.t:
                if t == 0:
                    self.model.continuity_cases.add(self.model.cases_held[dc,t] == start_capacities[dc])
                else:
                    const = self.model.cases_held[dc,t-1] 
                    try:
                        const -= sum(self.model.cases_dc_dem[dc, demand_node, t+transit_time_dc_dem[dc,demand_node]] for demand_node in self.model.demand_nodes) 
                    except:
                        pass
                    try:
                        const -= sum(self.model.cases_dc_dc[dc,dc2,t+transit_time_dc_dc[dc,dc2]] for dc2 in self.model.dcs)
                    except:
                        pass
                    try:
                        const += sum(self.model.cases_dc_dc[dc1, dc, t+transit_time_dc_dc[dc1,dc]] for dc1 in self.model.dcs)
                    except:
                        pass
                    try:
                        const += sum(self.model.cases_prod_dc[prod_node, dc, t+transit_time_prod_dc[prod_node,dc]] for prod_node in self.model.production_nodes)
                    except:
                        pass

                    self.model.continuity_cases.add(self.model.cases_held[dc,t] == const)

        def supply_rule(model, i,t):
            return sum(model.cases_dc_dem[i,j,t] for j in model.demand_nodes) <= model.cases_held[i,t]
        self.model.supply = Constraint(self.model.dcs, self.model.t, rule=supply_rule, doc='Observe supply limit at plant i')

        def production_rule(model, i, t):
            return model.cases_produced[i,t] <= production_capacity[i]
        self.model.produce = Constraint(self.model.production_nodes, self.model.t, doc='production capacities')

        def production_shipped_rule(model, i, t):
            return model.cases_produced[i, t] == sum(model.cases_prod_dc[i,dc,t] for dc in model.dcs) + 100
        self.model.production_continuity = Constraint(self.model.production_nodes, self.model.t, rule = production_shipped_rule, doc = 'production continuity')

        def demand_rule(model, j,t):
            return sum(model.cases_dc_dem[i,j,t] for i in model.dcs) >= model.demand[j,t] * 0.2
        self.model.demand_constr = Constraint(self.model.demand_nodes, self.model.t, rule=demand_rule, doc='Satisfy demand at market j')


        self.model.demand_costs = sum(self.model.c[i,j]*self.model.cases_dc_dem[i,j,t] for i in self.model.dcs for j in self.model.demand_nodes for t in self.model.t)
        self.model.dc_inter_costs = sum(self.model.c_dc[i,j]*self.model.cases_dc_dc[i,j,t] for i in self.model.dcs for j in self.model.dcs for t in self.model.t)
        self.model.storage_costs = sum(self.model.cases_held[dc,t] * self.model.storage_cost for dc in self.model.dcs for t in self.model.t)
        self.model.prod_trans_costs = sum(self.model.c_prod[i,j]*self.model.cases_prod_dc[i,j,t] for i in self.model.production_nodes for j in self.model.dcs for t in self.model.t)
        self.model.production_costs = sum(self.model.cases_produced[prod,t] * self.model.production_cost for prod in self.model.production_nodes for t in self.model.t)
        self.model.difference_in_demand = sum(sum(self.model.cases_dc_dem[i,j,t] for i in self.model.dcs) +  1*self.model.demand[j,t] for j in self.model.demand_nodes for t in self.model.t)

        def objective_rule(model):
            return model.demand_costs + model.storage_costs + model.dc_inter_costs + model.production_costs + model.prod_trans_costs + 1000000.0*model.difference_in_demand
        self.model.objective = Objective(rule=objective_rule, sense=minimize, doc='Define objective function')
        return self
    
    def solve(self):
        opt = SolverFactory("cbc", executable='C:\\Users\\Sam Cox\\Documents\\cbc\\Cbc-windeps-win64-msvc15-mtd\\bin\\cbc.exe')
        self.results = opt.solve(self.model)
        return self


        
        
if __name__ == '__main__':
    data = CovidData()
    data.get_county_cases()
    data.get_county_coordinates()
    trans_opt = TransOpt(data)
    trans_opt.create_model(trans_opt.data.df_cases)
    trans_opt.solve()
    trans_opt.model.objective.display()
    trans_opt.results.write()
    m= trans_opt.model
    
    print([[trans_opt.model.cases_produced[i, t].value for i in trans_opt.model.production_nodes] for t in trans_opt.model.t])
    print([[trans_opt.model.cases_held[i, t].value for i in trans_opt.model.dcs] for t in trans_opt.model.t])
    print(trans_opt.model.demand_costs(), trans_opt.model.storage_costs(), trans_opt.model.production_costs(), trans_opt.model.dc_inter_costs(), trans_opt.model.difference_in_demand())
    print([[[m.cases_prod_dc[prod,dc,t].value for prod in trans_opt.model.production_nodes] for dc in m.dcs] for t in m.t])