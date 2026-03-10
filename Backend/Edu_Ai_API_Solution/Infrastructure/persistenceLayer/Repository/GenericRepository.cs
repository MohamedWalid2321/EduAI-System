namespace persistenceLayer.Repository
{
	public class GenericRepository<Tentity, TKey>(LmsDBContext _context) : IGenericRepository<Tentity, TKey> where Tentity : BaseEntity<TKey>
	{
		public async Task AddAsync(Tentity entity)
		=> await _context.Set<Tentity>().AddAsync(entity);

		public void Delete(Tentity entity)
			=> _context.Set<Tentity>().Remove(entity);

		public async Task<IEnumerable<Tentity>> GetAllAsync()
			=> await Task.FromResult(_context.Set<Tentity>().AsEnumerable());

		public async Task<Tentity?> GetByIdAsync(TKey id)
			=> await _context.Set<Tentity>().FindAsync(id);

	

		public void Update(Tentity entity)
			=> _context.Set<Tentity>().Update(entity);
		public async Task<IEnumerable<Tentity>> GetAllAsync(ISpecifications<Tentity, TKey> specification)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).ToListAsync();
        public async Task<Tentity?> GetFirstOrDefaultAsync(ISpecifications<Tentity, TKey> specification)
        {
            var query = SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification);
            return await query.FirstOrDefaultAsync();
        }
        public async Task<Tentity?> GetByIdAsync(ISpecifications<Tentity, TKey> specification)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).FirstOrDefaultAsync();
		public async Task<int> GetCountAsync(ISpecifications<Tentity, TKey> specification)
			=> await SpecificationEvaluator.GetQuery(_context.Set<Tentity>().AsQueryable(), specification).CountAsync();
        
    }
}
