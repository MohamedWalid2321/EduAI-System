using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class addAcademicYearFees : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "AcademicYear",
                table: "AspNetUsers",
                newName: "AcademicYearEnum");

            migrationBuilder.AlterColumn<int>(
                name: "AcademicLevel",
                table: "Courses",
                type: "int",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "int");

            migrationBuilder.AddColumn<int>(
                name: "AcademicYearId",
                table: "AspNetUsers",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.CreateTable(
                name: "AcademicYear",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Name = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    CreatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AcademicYear", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Fee",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    AcademicYearId = table.Column<int>(type: "int", nullable: false),
                    Name = table.Column<int>(type: "int", maxLength: 100, nullable: false),
                    Amount = table.Column<decimal>(type: "decimal(10,2)", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    CreatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Fee", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Fee_AcademicYear_AcademicYearId",
                        column: x => x.AcademicYearId,
                        principalTable: "AcademicYear",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "Payments",
                columns: table => new
                {
                    Id = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Amount = table.Column<decimal>(type: "decimal(18,2)", nullable: false),
                    Currency = table.Column<string>(type: "nvarchar(max)", nullable: false),
                    Status = table.Column<int>(type: "int", nullable: false),
                    TransactionId = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    PaymentDate = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "GETUTCDATE()"),
                    PaymentMethod = table.Column<int>(type: "int", nullable: false),
                    StudentId = table.Column<string>(type: "nvarchar(450)", nullable: false),
                    AcademicYearId = table.Column<int>(type: "int", nullable: false),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: true, defaultValueSql: "GETUTCDATE()"),
                    LastUpdatedAt = table.Column<DateTime>(type: "datetime2", nullable: true),
                    CreatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true),
                    LastUpdatedBy = table.Column<string>(type: "nvarchar(max)", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Payments", x => x.Id);
                    table.ForeignKey(
                        name: "FK_Payments_AcademicYear_AcademicYearId",
                        column: x => x.AcademicYearId,
                        principalTable: "AcademicYear",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                    table.ForeignKey(
                        name: "FK_Payments_AspNetUsers_StudentId",
                        column: x => x.StudentId,
                        principalTable: "AspNetUsers",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                columns: new[] { "AcademicYearId", "PasswordHash" },
                values: new object[] { 0, "AQAAAAIAAYagAAAAEOWZVo54O7t+j0m6vtcBTTF9ZYE0F8PDMvFfkTBcLVpPrcU8SyXaYxD1uMxwvVvtWg==" });

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                columns: new[] { "AcademicYearId", "PasswordHash" },
                values: new object[] { 0, "AQAAAAIAAYagAAAAEMPNEtSHlXRLP5XwNknrW6yQcQlJG9m99gozXg3cangkZ8gfZN6m9AiXMAXETC2dGQ==" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                columns: new[] { "CreatedAt", "Title" },
                values: new object[] { new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7366), "ComputerEngineering" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7390));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7392));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7391));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 4, 24, 21, 55, 25, 307, DateTimeKind.Local).AddTicks(7392));

            migrationBuilder.CreateIndex(
                name: "IX_Fee_AcademicYearId_Name",
                table: "Fee",
                columns: new[] { "AcademicYearId", "Name" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_Payments_AcademicYearId",
                table: "Payments",
                column: "AcademicYearId");

            migrationBuilder.CreateIndex(
                name: "IX_Payments_StudentId",
                table: "Payments",
                column: "StudentId");

            migrationBuilder.CreateIndex(
                name: "IX_Payments_TransactionId",
                table: "Payments",
                column: "TransactionId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Fee");

            migrationBuilder.DropTable(
                name: "Payments");

            migrationBuilder.DropTable(
                name: "AcademicYear");

            migrationBuilder.DropColumn(
                name: "AcademicYearId",
                table: "AspNetUsers");

            migrationBuilder.RenameColumn(
                name: "AcademicYearEnum",
                table: "AspNetUsers",
                newName: "AcademicYear");

            migrationBuilder.AlterColumn<int>(
                name: "AcademicLevel",
                table: "Courses",
                type: "int",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "int",
                oldNullable: true);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEIiFB2MMRioICyCw9URGQjRe/VIWTH9jsdjA6r9ZE9b4fmizB4nwt0TeXjs/cF7ZsA==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEDIuDd/AXGh90C9pIbmvovNAYlCBW2t+f4+IHRn+wxIMztqXpG8xmUDz1Y1Dn4bmZw==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                columns: new[] { "CreatedAt", "Title" },
                values: new object[] { new DateTime(2026, 3, 6, 8, 14, 24, 535, DateTimeKind.Local).AddTicks(9755), "CommunicationEngineering" });

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 6, 8, 14, 24, 535, DateTimeKind.Local).AddTicks(9807));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 6, 8, 14, 24, 535, DateTimeKind.Local).AddTicks(9813));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 6, 8, 14, 24, 535, DateTimeKind.Local).AddTicks(9810));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 3, 6, 8, 14, 24, 535, DateTimeKind.Local).AddTicks(9812));
        }
    }
}
