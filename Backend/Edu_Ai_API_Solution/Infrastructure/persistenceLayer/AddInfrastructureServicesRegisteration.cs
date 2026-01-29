using DomainLayer.Contracts;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using persistenceLayer.Data;
using persistenceLayer.Repository;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer
{
	public static class AddInfrastructureServicesRegisteration
	{
		public static IServiceCollection AddInfrastructureServices(this IServiceCollection services,IConfiguration configuration)
		{
			services.AddDbContext<LmsDBContext>(options =>
			{
				options.UseSqlServer(configuration.GetConnectionString("DefaultConnection"));
			});
			services.AddScoped<IUnitOfWork, UnitOfWork>();
			return services;
		}
	}
}
