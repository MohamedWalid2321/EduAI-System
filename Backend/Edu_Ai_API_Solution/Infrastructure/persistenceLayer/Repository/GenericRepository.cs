namespace persistenceLayer.Repository
{
	public class GenericRepository<Tentity, TKey>(LmsDBContext _context) : IGenericRepository<Tentity, TKey> where Tentity : BaseEntity<TKey>
	{
		public async Task AddAsync(Tentity entity, CancellationToken cancellationToken = default)
			=> await _context.Set<Tentity>().AddAsync(entity, cancellationToken);

		public void Delete(Tentity entity)
		{
			entity.IsDeleted = true;
			entity.DeletedAt = DateTime.UtcNow;
			_context.Set<Tentity>().Update(entity);
		}

		/// <summary>
		/// Permanently removes the row from the database.
		/// Use instead of <see cref="Delete"/> when the entity has a unique index
		/// that does not filter on IsDeleted (e.g. UserCourses).
		/// </summary>
		public void HardDelete(Tentity entity)
			=> _context.Set<Tentity>().Remove(entity);

		public async Task<IEnumerable<Tentity>> GetAllAsync(CancellationToken cancellationToken = default)
			=> await _context.Set<Tentity>().ToListAsync(cancellationToken);

		public async Task<Tentity?> GetByIdAsync(TKey id, CancellationToken cancellationToken = default)
			=> await _context.Set<Tentity>().FindAsync(new object[] { id! }, cancellationToken);

		public void Update(Tentity entity)
			=> _context.Set<Tentity>().Update(entity);

		public async Task<IEnumerable<Tentity>> GetAllAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).ToListAsync(cancellationToken);

		public async Task<Tentity?> GetFirstOrDefaultAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default)
		{
			var query = SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification);
			return await query.FirstOrDefaultAsync(cancellationToken);
		}

		public async Task<Tentity?> GetByIdAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).FirstOrDefaultAsync(cancellationToken);

		public async Task<int> GetCountAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).CountAsync(cancellationToken);
	}
}
