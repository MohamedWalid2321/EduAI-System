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
	public class UnitOfWork(LmsDBContext _context) : IUnitOfWork
	{
		private readonly Dictionary<string, object> _repositories = [];
		public IGenericRepository<Tentity, TKey> GetRepository<Tentity, TKey>() where Tentity : BaseEntity<TKey>
		{
			var typeName = typeof(Tentity).Name;
			if (_repositories.ContainsKey(typeName))
			{
				return (IGenericRepository<Tentity, TKey>)_repositories[typeName]!;
			}
			var repositoryInstance = new GenericRepository<Tentity, TKey>(_context);
			_repositories.Add(typeName, repositoryInstance);
			return repositoryInstance;

		}

		public async Task<int> SaveChangesAsync()
		{
			return await _context.SaveChangesAsync();
		}
	}
}
