namespace persistenceLayer
{
	internal static class SpecificationEvaluator
	{
		public static IQueryable<Tentity> GetQuery<Tentity, TKey>(IQueryable<Tentity> inputQuery, ISpecifications<Tentity, TKey> spec) where Tentity : BaseEntity<TKey>
		{
			var query = inputQuery;
			if (spec.Criteria is not null)
			{
				query = query.Where(spec.Criteria);
			}
			if (spec.OrderBy is not null)
			{
				query = query.OrderBy(spec.OrderBy);
			}
			if (spec.OrderByDescending is not null)
			{
				query = query.OrderByDescending(spec.OrderByDescending);
			}
			if (spec.Includes is not null && spec.Includes.Count > 0)
			{
				query = spec.Includes.Aggregate(query, (current, include) => current.Include(include));
			}
			return query;
		}
	}
}
