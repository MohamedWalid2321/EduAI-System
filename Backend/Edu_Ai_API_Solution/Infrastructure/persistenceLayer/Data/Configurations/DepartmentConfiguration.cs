using Shared.Constants;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace persistenceLayer.Data.Configurations
{
	public class DepartmentConfiguration: IEntityTypeConfiguration<Department>
	{
		public void Configure(EntityTypeBuilder<Department> builder)
		{
			builder.HasData(
				new Department { Id = DefaultDepartment.ComputerEngineeringId, Title = DefaultDepartment.CommunicationEngineering },
				new Department { Id = DefaultDepartment.ElectricalEngineeringId, Title = DefaultDepartment.ElectricalEngineering },
				new Department { Id = DefaultDepartment.CommunicationEngineeringId, Title = DefaultDepartment.CommunicationEngineering },
				new Department { Id = DefaultDepartment.BiomedicalEngineeringId, Title = DefaultDepartment.BiomedicalEngineering },
				new Department { Id = DefaultDepartment.MechanicalEngineeringId, Title = DefaultDepartment.MechanicalEngineering }
			);
		}
	}
}	
