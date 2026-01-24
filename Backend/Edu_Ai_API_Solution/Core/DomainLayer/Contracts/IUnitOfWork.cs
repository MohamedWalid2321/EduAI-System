using DomainLayer.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Contracts
{
	public interface IUnitOfWork
	{
		public IGenericRepository<Tentity, TKey> GetRepository<Tentity, TKey>() where Tentity : BaseEntity<TKey>;
		Task<int> SaveChangesAsync();
	}
}
