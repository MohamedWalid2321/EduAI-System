using DomainLayer.Contracts;
using DomainLayer.Models;
using persistenceLayer.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
	}
}
