from .Range import Range
from . import RangeGeospatial, PinkNoise

import pandas as pd
import geopandas as gpd
import numpy as np

import shapely
import pyproj


class RangeCollection:
    def __init__(self):
        """
        Initialize a RangeCollection object.

        Attributes:
        Ranges (list): A list to hold Range instances.
        Abundances (list): A list to store the abundances of the ranges.
        name (str): Name of the collection (optional).
        time_points (dict): A dictionary to store time points (should be declared externally).
        edge_df (pd.DataFrame): A DataFrame to hold edge calculations.
        _MC_df (pd.DataFrame): A DataFrame for Monte Carlo estimates.
        _MC_edges_x (pd.DataFrame): A DataFrame for edge x-values with Monte Carlo estimates.
        _MC_edges_y (pd.DataFrame): A DataFrame for edge y-values with Monte Carlo estimates.
        _MC_Abundances (pd.DataFrame): A DataFrame for Monte Carlo estimates of abundances.
        _MC_compatibleTrends (pd.DataFrame): A DataFrame for Monte Carlo-compatible trends.

        """

        self.Ranges = []
        self.Abundances = []

        # Need to be declared externally
        self.name = None
        self.time_points = dict()

        # Hidden Attributes to hold as placeholders
        self.edge_df = None
        self._MC_df = None
        self._MC_edges_x = None
        self._MC_edges_y = None
        self._MC_Abundances = None
        self._MC_compatibleTrends = None

    def append(self, new_range):
        """
        Append a new Range instance to the collection.

        Parameters:
        new_range (Range): A Range instance to be added to the collection.

        """
        if isinstance(new_range, Range):
            self.Ranges.append(new_range)
            self.Abundances.append(new_range.Abundance)
        else:
            raise ValueError("Input must be a Range instance")

    def __getitem__(self, slice_obj):
        """
        Retrieve a slice of Range instances from the collection.

        Parameters:
        slice_obj (slice or int): A slice or index to select Range instances.

        Returns:
        Range or RangeCollection: A single Range or a new RangeCollection containing a slice of Range instances.

        """
        if isinstance(slice_obj, slice):
            new_collection = RangeCollection()
            new_collection.Ranges = self.Ranges[slice_obj]
            new_collection.edge_df = self.edge_df # THE WHOLE edge_df is picked up
            return new_collection
        else:
            return self.Ranges[slice_obj]

    def __len__(self):
        """
        Get the number of Range instances in the collection.

        Returns:
        int: The number of Range instances.

        """
        return len(self.Ranges)

    def reproject(self, target_crs):
        """
        Reproject the RangeCollection to the specified CRS, affecting only CRS-dependent attributes.

        Parameters:
        target_crs (pyproj.crs.CRS): The target CRS for reprojection.

        """

        # Reproject calcuated edge_df
        if self.edge_df is not None:
          for range_obj in self.Ranges:


            ## define reprojector
            source_crs = range_obj.crs
            reprojection = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform

            def func(x):
              return shapely.ops.transform(reprojection,x)

            self.edge_df[range_obj.name] = self.edge_df[range_obj.name].apply(func)

        # Reproject ranges
        for range_obj in self.Ranges:
            range_obj.reproject(target_crs)

    @property
    def centroids(self):
        """
        Get the centroids of the Range instances.

        Returns:
        list: A list of centroid points.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.centroid for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    @property
    def naive_medians(self):
        """
        Get the naive_medians of the Range instances.

        Returns:
        list: A list of naive_medians points.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.naive_median for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    
    @property
    def geometric_medians(self):
        """
        Get the geometric medians of the Range instances.

        Returns:
        list: A list of geometric median points.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.geometric_median for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    @property
    def densities(self):
        """
        Get the densities of the Range instances.

        Returns:
        list: A list of density values.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.density for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    @property
    def areas(self):
        """
        Get the areas of the Range instances.

        Returns:
        list: A list of area values.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [np.sum(range_obj.area)/10**6 for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")


    @property
    def concave_hulls(self):
        """
        Get the concave hulls of the Range instances.

        Returns:
        list: A list of concave hull polygons.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.concave_hull for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    @property
    def convex_hulls(self):
        """
        Get the convex hulls of the Range instances.

        Returns:
        list: A list of convex hull polygons.

        """
        if all(range_obj.crs == self.Ranges[0].crs for range_obj in self.Ranges):
            return [range_obj.convex_hull for range_obj in self.Ranges]
        else:
            raise ValueError("Not all items in the collection have the same CRS. Use RangeCollection.reproject() to synchronize them")

    def calc_edges(self, edges, alpha_values):
        """
        Calculate edge values for a range of alphas and edges.

        Parameters:
        edges (list): A list of edge names (e.g., 'N', 'S', 'E', 'W').
        alpha_values (list): A list of alpha values.

        Returns:
        pd.DataFrame: A DataFrame containing edge calculations.

        """
        iterables=[edges,np.round(alpha_values,2)]
        index = pd.MultiIndex.from_product(iterables,names=['edge','alpha'])
        df = pd.DataFrame(index=index)

        for range_obj in self.Ranges:
          df[range_obj.name] = [range_obj.calc_edge_geometric_median(e,a) for e,a in df.index ]

        self.edge_df = df

        return self.edge_df

    @property
    def edges_x(self):
        """
        Get the x-values from edge calculations.

        Returns:
        pd.DataFrame: A DataFrame with x-values.

        """
        def func(z):
          return z.x
        return self.edge_df.applymap(func)

    @property
    def edges_y(self):
        """
        Get the y-values from edge calculations.

        Returns:
        pd.DataFrame: A DataFrame with y-values.

        """
        def func(z):
          return z.y
        return self.edge_df.applymap(func)

    def _MC_estimates_row(self, array, msg_info=None):
        """
        Construct a row (dictionary) for Monte Carlo estimates.

        Parameters:
        array (numpy.ndarray): An array for Monte Carlo estimation.

        Returns:
        dict: A dictionary with Monte Carlo estimates.

        """
        row = {}

        # Calculate time differences relative to the initial time point (t=0)
        time = self.time_points['mid']
        #time = [x - time[0] for x in time]
        xy_array = np.array([time,array])
        t_max = len(time)

        # White Noise
        MC = PinkNoise.MonteCarlo_significance(xy_array,MC_reps=10**3,
                                               noise_func= PinkNoise.noise_white, noise_kwargs = {'tmax':t_max},
                                               log_kwargs={'name':self.name,'info':msg_info})

        row['Slope_(white)'] = MC[1]
        row['Intercept_(white)'] = MC[2]
        row['White_Significance_MC'] = MC[0]
        row['White_Significance_fit'] = PinkNoise.white_fit_significance(xy_array)
        row['White_signfc_deviation'] = row['White_Significance_fit'] - row['White_Significance_MC']



        # Pink Noise
        MC = PinkNoise.MonteCarlo_significance(xy_array,MC_reps=10**3,
                                               noise_func = PinkNoise.noise_pink, noise_kwargs = {'nu':1,'tmax':t_max,'beta':2},
                                               log_kwargs={'name':self.name})


        row['Slope_(pink)'] = MC[1]
        row['Intercept_(pink)'] = MC[2]
        row['Pink_Significance_MC'] = MC[0]
        row['Pink_Significance_fit'] = np.nan    #pink_fit_significance(df)
        row['Pink_signfc_deviation'] = np.nan  #row['Pink_Significance_fit'] - row['Pink_Significance_MC']

        return row

    @property
    def MC_df(self):
        """
        Get Monte Carlo estimates for slope significance and other quantities.

        Returns:
        pd.DataFrame: A DataFrame with Monte Carlo estimates.

        """
        if self._MC_df is None:

          # Create a ditionary of the variables that need computing
          diction = {'Area(km^2)':self.areas,
                    'Density':self.densities,
                    'Centroid.y':[z.y for z in self.centroids],
                    'Centroid.x':[z.x for z in self.centroids],
                    'Geom_median.y':[z.y for z in self.geometric_medians],
                    'Geom_median.x':[z.x for z in self.geometric_medians]}

          table = pd.DataFrame()

          for name,array in diction.items():
            # Add row to table
            row = self._MC_estimates_row(array,msg_info=name)
            row_df = pd.DataFrame(row, index=[name])
            table = pd.concat([table, row_df], ignore_index=False)

          self._MC_df = table

        return self._MC_df

    @property
    def MC_edges_x(self):
        """
        Get edge x-values with Monte Carlo estimates.

        Returns:
        pd.DataFrame: A DataFrame with x-values and Monte Carlo estimates.

        """
        if self._MC_edges_x is None:
          table = pd.DataFrame()

          for idx,row_values in self.edges_x.iterrows():
              row = self._MC_estimates_row(np.array(row_values),msg_info=f'x_edge_alpha={idx}')
              row_df = pd.DataFrame(row, index=[idx])
              table = pd.concat([table, row_df], ignore_index=False)

          table.index = pd.MultiIndex.from_tuples(table.index.tolist(),names=self.edge_df.index.names)
          self._MC_edges_x = table

        return self._MC_edges_x

    @property
    def MC_edges_y(self):
        """
        Get edge y-values with Monte Carlo estimates.

        Returns:
        pd.DataFrame: A DataFrame with y-values and Monte Carlo estimates.

        """
        if self._MC_edges_y is None:
          table = pd.DataFrame()

          for idx,row_values in self.edges_x.iterrows():
              row = self._MC_estimates_row(np.array(row_values),msg_info=f'y_edge_alpha={idx}')
              row_df = pd.DataFrame(row, index=[idx])
              table = pd.concat([table, row_df], ignore_index=False)

          table.index = pd.MultiIndex.from_tuples(table.index.tolist(),names=self.edge_df.index.names)
          self._MC_edges_y = table

        return self._MC_edges_y

    @property
    def MC_Abundances(self):
        """
        Get Monte Carlo estimates for abundances.

        Returns:
        pd.DataFrame: A DataFrame with Monte Carlo estimates for abundances.

        """
        if self._MC_Abundances is None:
            table = pd.DataFrame(self._MC_estimates_row(np.array(self.Abundances),msg_info=f'abundance'),index=['Abundance'])

            self._MC_Abundances = table
        return self._MC_Abundances

    @property
    def MC_compatibleTrends(self):
        """
        Get Monte Carlo-compatible trends.

        Returns:
        pd.DataFrame: A DataFrame with Monte Carlo-compatible trends.

        """
        if self._MC_compatibleTrends is None:
            table = pd.DataFrame()

            # Calculate time differences relative to the initial time point (t=0)
            time = self.time_points['mid']
            #time = [x - time[0] for x in time]
            xy_array = np.array([time,self.Abundances])
            t_max = len(time)

            n_trends = 100

            # white noise
            slopes_w, intercepts_w = PinkNoise.MonteCarlo_compatibleTrends(xy_array,fitted_slope=float(self.MC_Abundances['Slope_(white)']),
                                                            noise_func= PinkNoise.noise_white, noise_kwargs = {'tmax':t_max},n_trends=n_trends )

            # pink noise
            slopes_p, intercepts_p = PinkNoise.MonteCarlo_compatibleTrends(xy_array,fitted_slope=float(self.MC_Abundances['Slope_(pink)']),
                                                            noise_func = PinkNoise.noise_pink, noise_kwargs = {'nu':1,'tmax':t_max,'beta':2},n_trends=n_trends )


            data = pd.DataFrame({'Slope(white)': slopes_w,'Intercept(white)': intercepts_w,
                                 'Slope(pink)': slopes_p,'Intercept(pink)': intercepts_p})
            table = pd.concat([table, data], ignore_index=True)


            self._MC_compatibleTrends = table
        return self._MC_compatibleTrends
