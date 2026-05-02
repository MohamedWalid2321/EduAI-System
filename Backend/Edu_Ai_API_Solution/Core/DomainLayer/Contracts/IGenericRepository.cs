using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Contracts
{
	public interface IGenericRepository<Tentity, TKey> where Tentity : BaseEntity<TKey>
	{
		Task<Tentity?> GetByIdAsync(TKey id, CancellationToken cancellationToken = default);
		Task<IEnumerable<Tentity>> GetAllAsync(CancellationToken cancellationToken = default);
		Task AddAsync(Tentity entity, CancellationToken cancellationToken = default);
		void Update(Tentity entity);
		void Delete(Tentity entity);
		// With Specifications
		Task<Tentity?> GetByIdAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default);
		Task<IEnumerable<Tentity>> GetAllAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default);
		Task<Tentity?> GetFirstOrDefaultAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default);
		Task<int> GetCountAsync(ISpecifications<Tentity, TKey> specification, CancellationToken cancellationToken = default);
	}
}
