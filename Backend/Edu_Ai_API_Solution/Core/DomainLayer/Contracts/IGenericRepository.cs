using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Contracts
{
	public interface IGenericRepository<Tentity,TKey> where Tentity : BaseEntity<TKey>
	{
		Task<Tentity?> GetByIdAsync(TKey id);
		Task<IEnumerable<Tentity>> GetAllAsync();
		Task AddAsync(Tentity entity);
		void Update(Tentity entity);
		void Delete(Tentity entity);
		//With Specifications
		Task<Tentity?> GetByIdAsync(ISpecifications<Tentity, TKey> specification);
		Task<IEnumerable<Tentity>> GetAllAsync(ISpecifications<Tentity, TKey> specification);
		Task<int> GetCountAsync(ISpecifications<Tentity, TKey> specification);
		

    }
}
